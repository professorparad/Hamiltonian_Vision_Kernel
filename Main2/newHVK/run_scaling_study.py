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


def read_results_with_retry(path: Path, attempts: int = 5, delay_s: float = 1.0) -> dict:
    """A second, unrelated process on this machine has repeatedly (and, as
    far as we can tell, accidentally) re-launched this exact script against
    this exact results file, racing our own read/write cycle and crashing
    the orchestrator with a JSONDecodeError when it reads mid-write. Retry
    a transient bad read a few times before giving up, instead of dying."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(delay_s)
    print(f"WARNING: could not read {path} after {attempts} attempts ({last_error}); "
          f"starting from an empty in-memory results dict for this run", flush=True)
    return {"qubit_sweep": [], "bond_dim_sweep": [], "depth_sweep": []}


def write_results_atomic(path: Path, results: dict) -> None:
    """Write-then-rename instead of an in-place write_text, so a concurrent
    reader never observes a half-written file (os.replace is atomic on the
    same filesystem on both POSIX and Windows)."""
    import os

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str))
    os.replace(tmp, path)


def main(smoke_test: bool = False):
    global STEPS, SEEDS, RESULT_FILE
    if smoke_test:
        STEPS = 2
        SEEDS = [0]
        RESULT_FILE = OUT_DIR / "scaling_study_SMOKE_TEST.json"  # never touch the production results file

    if RESULT_FILE.exists():
        results = read_results_with_retry(RESULT_FILE)
    else:
        results = {"qubit_sweep": [], "bond_dim_sweep": [], "depth_sweep": []}

    def already_done(bucket: str, config: dict) -> bool:
        for r in results[bucket]:
            if r.get("status") in ("failed", "timed_out"):
                continue
            if all(r.get(k) == v for k, v in config.items()):
                return True
        return False

    pending: list[tuple[str, dict]] = []
    for q in ([4, 6, 8] if not smoke_test else [4]):
        for seed in SEEDS:
            config = {"topology": "HVK1D", "seed": seed, "qubit_count": q, "bond_dim": 4, "steps": STEPS}
            if not already_done("qubit_sweep", config):
                pending.append(("qubit_sweep", config))
    for topology in ("HVK1D", "HVK2D"):
        for chi in ([1, 2, 4, 8] if not smoke_test else [1]):
            for seed in SEEDS:
                config = {"topology": topology, "seed": seed, "qubit_count": 6, "bond_dim": chi, "steps": STEPS}
                if not already_done("bond_dim_sweep", config):
                    pending.append(("bond_dim_sweep", config))
    for topology in ("HVK1D", "HVK2D"):
        for depth in ([1, 2, 3, 4] if not smoke_test else [1]):
            for seed in SEEDS:
                config = {"topology": topology, "seed": seed, "qubit_count": 6, "bond_dim": 4, "n_layers": depth, "steps": STEPS}
                if not already_done("depth_sweep", config):
                    pending.append(("depth_sweep", config))

    print(f"=== {len(pending)} configs remaining, running with {MAX_WORKERS} concurrent workers ===", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    write_lock = threading.Lock()

    def do_one(bucket_config):
        bucket, config = bucket_config
        return bucket, run_isolated(config)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(do_one, bc) for bc in pending]
        for i, future in enumerate(as_completed(futures), start=1):
            bucket, r = future.result()
            print(f"[{i}/{len(pending)}] {json.dumps(r)}", flush=True)
            with write_lock:
                results[bucket].append(r)
                write_results_atomic(RESULT_FILE, results)

    print("\nDone. Saved to", RESULT_FILE, flush=True)


if __name__ == "__main__":
    if "--single-run" in sys.argv:
        idx = sys.argv.index("--single-run")
        run_single_from_cli(json.loads(sys.argv[idx + 1]))
    else:
        smoke = "--smoke-test" in sys.argv
        main(smoke_test=smoke)
