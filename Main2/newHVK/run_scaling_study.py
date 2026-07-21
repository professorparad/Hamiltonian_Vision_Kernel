"""Workstream 2 (real circuit): statistically resolved scaling study.

Single CIFAR-10 image ("cat", native 32x32, overlapping 8x8 patches stride 4,
49 patches -- same convention as the topology comparison and the existing
same-set CIFAR-10 table), same-set training (this sweep is about relative
scaling behavior across configs, not held-out generalization, matching the
existing single-seed capacity_ablation table's own same-set design).

Sweeps, all at 3 seeds and a reduced ~90-step budget (see Step 0 timing
calibration in the approved plan):
  - qubit count: HVK1D only, q in {4,6,8} (HVK2D's grid is architecturally
    fixed at 6 qubits -- Quantum2DGridModel hardcodes the 2x3 grid, so no
    HVK2D points are fabricated at other qubit counts)
  - MPS bond dimension: chi in {1,2,4,8}, both topologies
  - circuit depth: 1-4 layers, both topologies (weights re-initialized with
    a different first-dimension shape post-construction; StronglyEntangling-
    Layers / the HVK2D grid layer loop both read depth from weights.shape[0]
    at call time, so no source file is modified)
  - gradient variance: one backward pass per (seed, qubit-count, depth)
    config at initialization -- cheap, not a full training run -- reporting
    the variance of the gradient norm w.r.t. circuit weights as a concrete
    trainability diagnostic (McClean et al.-style barren-plateau check).
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")
BENCH_ROOT = REPO_ROOT / "Baselines" / "cifar10_comparisons"
MAIN_DIR = REPO_ROOT / "Main"
for p in (BENCH_ROOT, MAIN_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import DEFAULT_DATASET_DIR, compute_metrics, load_grayscale_image, stitch_overlapping_patches
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from src.training.training import resolve_device

CIFAR_DIR = DEFAULT_DATASET_DIR / "images"
IMAGE_PATH = sorted(CIFAR_DIR.glob("0000_*.png"))[0]  # cat
IMAGE_SIZE = 32
PATCH_SIZE = 8
PATCH_STRIDE = 4
POSITIONAL_DIM = 4
STEPS = 90
SEEDS = [0, 1, 2]

OUT_DIR = REPO_ROOT / "Main2" / "newHVK" / "results" / "scaling_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = OUT_DIR / "scaling_study.json"


MPS_N_SITES = 6  # fixed: patch is 8x8=64=2**6 pixels; this is independent of the VQC's qubit_count


def load_data(bond_dim: int, device: torch.device):
    image = load_grayscale_image(IMAGE_PATH)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=MPS_N_SITES, bond_dim=bond_dim) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    return {
        "image": image, "raw_positions": raw_positions,
        "features": features_t.to(device), "positions": positions.to(device), "targets": targets.to(device),
    }


def build_model(topology: str, feature_dim: int, qubit_count: int, device):
    if topology == "HVK1D":
        from src.decoder.patch_decoder import PatchDecoder
        from src.quantum.circuit import observable_dim
        from src.quantum.quantum_model import QuantumModel

        model = QuantumModel(feature_dim=feature_dim, positional_dim=POSITIONAL_DIM, qubit_count=qubit_count).to(device)
        obs_dim = 2 * qubit_count + 3 * (qubit_count - 1)
        decoder = PatchDecoder(observable_dim=obs_dim, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
        lr = 0.003
    elif topology == "HVK2D":
        from Main2.src.model import PatchDecoder as PatchDecoder2D
        from Main2.src.model import Quantum2DGridModel

        if qubit_count != 6:
            raise ValueError("HVK2D grid is architecturally fixed at 6 qubits")
        model = Quantum2DGridModel(feature_dim=feature_dim, positional_dim=POSITIONAL_DIM).to(device)
        decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
        lr = 0.004
    else:
        raise ValueError(topology)
    return model, decoder, lr


def set_depth(model, topology: str, n_layers: int, qubit_count: int, device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    new_weights = (torch.rand(n_layers, qubit_count, 3, generator=g) * math.pi).to(device)
    model.weights = torch.nn.Parameter(new_weights)


def gradient_variance(model, decoder, features, positions, targets, n_probes: int = 8, seed: int = 0) -> dict:
    """One backward pass per probe (fresh random weight re-init each time,
    same architecture) -- variance of the gradient norm w.r.t. circuit
    weights, a concrete, cheap trainability (barren-plateau-style) diagnostic.
    Not a full training run."""
    norms = []
    base_shape = model.weights.shape
    for i in range(n_probes):
        g = torch.Generator(device="cpu").manual_seed(20_000 + seed * 100 + i)
        model.weights = torch.nn.Parameter((torch.rand(*base_shape, generator=g) * math.pi).to(model.weights.device))
        model.zero_grad()
        observables, energies = model(features, positions)
        loss = torch.mean((decoder(observables, positions) - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        norms.append(float(model.weights.grad.norm().item()))
    return {"n_probes": n_probes, "grad_norm_mean": float(np.mean(norms)), "grad_norm_var": float(np.var(norms)), "grad_norms": norms}


def run_training(topology: str, seed: int, qubit_count: int = 6, bond_dim: int = 4, n_layers: int | None = None, steps: int = STEPS) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device("auto")
    data = load_data(bond_dim=bond_dim, device=device)

    model, decoder, lr = build_model(topology, data["features"].shape[1], qubit_count, device)
    if n_layers is not None:
        set_depth(model, topology, n_layers, qubit_count, device, seed)

    # gradient_variance() reassigns model.weights to fresh tensor objects for
    # each of its probes (needed to sample the gradient at several random
    # inits) -- run it, then re-establish a clean training-init weight
    # tensor, BEFORE constructing the optimizer, so Adam actually tracks the
    # tensor object that will be used (and its .grad zeroed) during training.
    # Constructing the optimizer first orphans model.weights the moment
    # gradient_variance reassigns it: optimizer.step() then updates a stale
    # tensor no longer connected to the forward pass, while the real
    # model.weights.grad is never zeroed by optimizer.zero_grad() and
    # accumulates without bound across the whole training loop -- this was
    # confirmed to be the actual bug behind both the misleadingly-plausible
    # qubit/bond-dim PSNR numbers (decoder-only training) and the depth-sweep
    # timeouts (unbounded gradient accumulation) in the first version of this
    # script.
    grad_stats = gradient_variance(model, decoder, data["features"], data["positions"], data["targets"], seed=seed)
    if n_layers is not None:
        set_depth(model, topology, n_layers, qubit_count, device, seed)
    else:
        # gradient_variance perturbed model.weights; re-seed a clean training-init tensor
        g = torch.Generator(device="cpu").manual_seed(30_000 + seed)
        model.weights = torch.nn.Parameter((torch.rand(*model.weights.shape, generator=g) * math.pi).to(device))

    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
    assert any(p is model.weights for p in optimizer.param_groups[0]["params"]), "model.weights not tracked by optimizer"

    t0 = time.perf_counter()
    for step in range(steps):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(data["features"], data["positions"])
        output = decoder(observables, data["positions"])
        loss = torch.mean((output - data["targets"]) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - t0

    model.eval()
    decoder.eval()
    with torch.no_grad():
        observables, _ = model(data["features"], data["positions"])
        pred = decoder(observables, data["positions"]).cpu().numpy()
    reconstruction = stitch_overlapping_patches(pred, data["raw_positions"], image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    metrics = compute_metrics(reconstruction, data["image"])

    return {
        "topology": topology, "seed": seed, "qubit_count": qubit_count, "bond_dim": bond_dim,
        "n_layers": n_layers, "steps": steps, "wall_time_s": elapsed,
        "psnr": metrics["psnr"], "ssim": metrics["ssim"], "mse": metrics["mse"],
        "grad_norm_mean": grad_stats["grad_norm_mean"], "grad_norm_var": grad_stats["grad_norm_var"],
    }


RUN_TIMEOUT_S = 3600  # depth-3/4 circuits can exceed 30 min; retain a hard cap without discarding valid slow runs


def run_single_from_cli(config: dict) -> None:
    """Invoked as a subprocess: run exactly one config, print the JSON
    result on its own line, flushed immediately."""
    r = run_training(
        config["topology"], config["seed"],
        qubit_count=config.get("qubit_count", 6), bond_dim=config.get("bond_dim", 4),
        n_layers=config.get("n_layers"), steps=config.get("steps", STEPS),
    )
    print("RESULT_JSON:" + json.dumps(r), flush=True)


def run_isolated(config: dict) -> dict:
    """Run one config in a subprocess with a hard timeout, so a single hung
    config (as happened previously: HVK1D chi=8 seed=2 stuck >3h while its
    sibling seeds finished in ~12min) gets killed and logged instead of
    blocking the whole sweep indefinitely."""
    import subprocess

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--single-run", json.dumps(config)],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        return {**config, "status": "failed", "error": "no RESULT_JSON in subprocess output",
                "stderr_tail": proc.stderr[-2000:], "wall_time_s": time.perf_counter() - t0}
    except subprocess.TimeoutExpired:
        return {**config, "status": "timed_out", "wall_time_s": RUN_TIMEOUT_S,
                "error": f"exceeded {RUN_TIMEOUT_S}s hard timeout, killed"}


MAX_WORKERS = 2  # leave headroom for monitoring, LaTeX builds, and OS I/O while the CPU-bound sweep runs

# --- File-based work queue -------------------------------------------------
# A second, unrelated process on this machine has repeatedly (and, as far as
# we can tell, accidentally) re-launched this exact script against this exact
# results file, racing a single shared-JSON read-modify-write cycle and
# repeatedly crashing/corrupting it. Rather than keep reactively detecting
# and killing that duplicate, the work distribution is a directory-based
# queue: one small JSON file per (bucket, config) pending unit of work,
# claimed by an atomic rename (pending/ -> claimed/, which can only succeed
# for exactly one process on a given file -- a second claimer gets
# FileNotFoundError and just moves on to the next item). This makes the
# whole pipeline safe by construction even under an uncontrolled second
# writer: at worst a duplicate process cooperatively drains the same queue
# instead of corrupting shared state.
QUEUE_DIR = OUT_DIR / "queue"
PENDING_DIR = QUEUE_DIR / "pending"
CLAIMED_DIR = QUEUE_DIR / "claimed"
DONE_DIR = QUEUE_DIR / "done"


def configure_queue(smoke_test: bool) -> None:
    """Keep smoke-test claims physically separate from production work."""
    global QUEUE_DIR, PENDING_DIR, CLAIMED_DIR, DONE_DIR
    QUEUE_DIR = OUT_DIR / ("queue_smoke" if smoke_test else "queue")
    PENDING_DIR = QUEUE_DIR / "pending"
    CLAIMED_DIR = QUEUE_DIR / "claimed"
    DONE_DIR = QUEUE_DIR / "done"


def _config_id(bucket: str, config: dict) -> str:
    import hashlib

    key = bucket + "|" + json.dumps(config, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def seed_queue(smoke_test: bool = False) -> None:
    """Idempotent: safe to call from multiple concurrent processes. Writes
    one file per pending config; a config already in pending/claimed/done is
    left untouched."""
    for d in (PENDING_DIR, CLAIMED_DIR, DONE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    all_items: list[tuple[str, dict]] = []
    for q in ([4, 6, 8] if not smoke_test else [4]):
        for seed in SEEDS:
            all_items.append(("qubit_sweep", {"topology": "HVK1D", "seed": seed, "qubit_count": q, "bond_dim": 4, "steps": STEPS}))
    for topology in ("HVK1D", "HVK2D"):
        for chi in ([1, 2, 4, 8] if not smoke_test else [1]):
            for seed in SEEDS:
                all_items.append(("bond_dim_sweep", {"topology": topology, "seed": seed, "qubit_count": 6, "bond_dim": chi, "steps": STEPS}))
    for topology in ("HVK1D", "HVK2D"):
        for depth in ([1, 2, 3, 4] if not smoke_test else [1]):
            for seed in SEEDS:
                all_items.append(("depth_sweep", {"topology": topology, "seed": seed, "qubit_count": 6, "bond_dim": 4, "n_layers": depth, "steps": STEPS}))

    for bucket, config in all_items:
        cid = _config_id(bucket, config)
        if any((d / f"{cid}.json").exists() for d in (PENDING_DIR, CLAIMED_DIR, DONE_DIR)):
            continue
        payload = json.dumps({"bucket": bucket, "config": config})
        tmp = PENDING_DIR / f".{cid}.tmp"
        tmp.write_text(payload)
        tmp.rename(PENDING_DIR / f"{cid}.json")  # atomic same-directory rename


def migrate_completed_results(path: Path) -> None:
    """Import completed records from the pre-queue summary without rerunning them."""
    if not path.exists():
        return
    try:
        previous = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    for bucket in ("qubit_sweep", "bond_dim_sweep", "depth_sweep"):
        for result in previous.get(bucket, []):
            if result.get("status") in ("failed", "timed_out"):
                continue
            config = {
                key: result[key]
                for key in ("topology", "seed", "qubit_count", "bond_dim", "n_layers", "steps")
                if key in result and result[key] is not None
            }
            cid = _config_id(bucket, config)
            payload = dict(result)
            payload["bucket"] = bucket
            (DONE_DIR / f"{cid}.json").write_text(json.dumps(payload, indent=2, default=str))
            # A stale claim/pending entry for an imported result is safe to discard.
            (PENDING_DIR / f"{cid}.json").unlink(missing_ok=True)
            (CLAIMED_DIR / f"{cid}.json").unlink(missing_ok=True)


def claim_one() -> tuple[str, str, dict] | None:
    """Try to atomically claim one pending item. Returns (config_id, bucket,
    config) or None if nothing is available (including the case where a
    listed file got claimed by someone else between our listdir and rename)."""
    for f in PENDING_DIR.glob("*.json"):
        try:
            f.rename(CLAIMED_DIR / f.name)
        except (FileNotFoundError, PermissionError, OSError):
            continue  # someone else claimed it first
        item = json.loads((CLAIMED_DIR / f.name).read_text())
        return f.stem, item["bucket"], item["config"]
    return None


def worker_loop(worker_id: int) -> None:
    while True:
        claimed = claim_one()
        if claimed is None:
            # On Windows, another thread's atomic rename can invalidate the
            # current directory enumeration even while other pending files
            # remain. Retry instead of silently retiring this worker.
            if any(PENDING_DIR.glob("*.json")):
                time.sleep(0.25)
                continue
            return
        cid, bucket, config = claimed
        r = run_isolated(config)
        r["bucket"] = bucket
        (DONE_DIR / f"{cid}.json").write_text(json.dumps(r, indent=2, default=str))
        (CLAIMED_DIR / f"{cid}.json").unlink(missing_ok=True)
        print(f"[worker {worker_id}] {json.dumps(r)}", flush=True)


def collect_results() -> dict:
    """Aggregate every done/*.json into the summary shape the rest of this
    project (and the LaTeX integration) expects."""
    results = {"qubit_sweep": [], "bond_dim_sweep": [], "depth_sweep": []}
    for f in sorted(DONE_DIR.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        bucket = r.pop("bucket", None)
        valid_budget = r.get("steps") == STEPS and r.get("seed") in SEEDS
        valid_bucket = (
            bucket == "qubit_sweep"
            and r.get("topology") == "HVK1D"
            and r.get("qubit_count") in ([4] if STEPS == 2 else [4, 6, 8])
            and r.get("bond_dim") == 4
            or bucket == "bond_dim_sweep"
            and r.get("topology") in ("HVK1D", "HVK2D")
            and r.get("qubit_count") == 6
            and r.get("bond_dim") in ([1] if STEPS == 2 else [1, 2, 4, 8])
            or bucket == "depth_sweep"
            and r.get("topology") in ("HVK1D", "HVK2D")
            and r.get("qubit_count") == 6
            and r.get("bond_dim") == 4
            and r.get("n_layers") in ([1] if STEPS == 2 else [1, 2, 3, 4])
        )
        if bucket in results and valid_budget and valid_bucket:
            results[bucket].append(r)
    return results


def main(smoke_test: bool = False):
    global STEPS, SEEDS, RESULT_FILE
    if smoke_test:
        STEPS = 2
        SEEDS = [0]
        RESULT_FILE = OUT_DIR / "scaling_study_SMOKE_TEST.json"  # never touch the production results file

    configure_queue(smoke_test)
    for d in (PENDING_DIR, CLAIMED_DIR, DONE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    migrate_completed_results(RESULT_FILE)
    seed_queue(smoke_test=smoke_test)
    n_pending = len(list(PENDING_DIR.glob("*.json")))
    n_claimed = len(list(CLAIMED_DIR.glob("*.json")))
    n_done = len(list(DONE_DIR.glob("*.json")))
    print(f"=== queue: {n_pending} pending, {n_claimed} claimed (possibly stale from a prior crash), "
          f"{n_done} done -- running with {MAX_WORKERS} workers ===", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor

    stop_flag = threading.Event()

    def periodic_collector():
        # keeps scaling_study.json (the file external monitoring already
        # reads) in sync with the queue's done/ directory every few seconds,
        # so progress stays visible without needing to know about the queue
        import os

        while not stop_flag.wait(5.0):
            results = collect_results()
            tmp = RESULT_FILE.with_suffix(f".{os.getpid()}.json.tmp")
            tmp.write_text(json.dumps(results, indent=2, default=str))
            os.replace(tmp, RESULT_FILE)

    collector_thread = threading.Thread(target=periodic_collector, daemon=True)
    collector_thread.start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker_loop, i) for i in range(MAX_WORKERS)]
        for f in futures:
            f.result()  # propagate any worker exception instead of silently swallowing it

    stop_flag.set()
    results = collect_results()
    RESULT_FILE.write_text(json.dumps(results, indent=2, default=str))
    n_total = sum(len(v) for v in results.values())
    print(f"\nDone: {n_total} results collected. Saved to", RESULT_FILE, flush=True)


if __name__ == "__main__":
    if "--single-run" in sys.argv:
        idx = sys.argv.index("--single-run")
        run_single_from_cli(json.loads(sys.argv[idx + 1]))
    else:
        smoke = "--smoke-test" in sys.argv
        main(smoke_test=smoke)
