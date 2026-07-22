"""Workstream 1 (real-circuit confirmation): HVK1D vs HVK2D topology comparison.

Trains each topology jointly on 2 CIFAR-10 images (native 32x32, overlapping
8x8 patches stride 4 -> 49 patches/image, matching the exact convention
already used for the same-set CIFAR-10 table in supplementary_study.tex) for
a reduced ~90-step budget across 3 seeds, then evaluates held-out
reconstruction on 3 further CIFAR-10 images from different classes never
seen in training -- a genuine held-out test, not same-set memorization.

This is the real-trained-circuit confirmation subset; the full-scope
surrogate sweep (5 seeds x 5 splits x 3 datasets) is a separate script.
Circuit depth / 2-qubit gate counts are not recomputed here -- they are
already reported, transpiled and verified, in Table (ibm_circuit_summary)
of the hardware-pilot methodology section.
"""
from __future__ import annotations

import json
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
IMAGE_SIZE = 32
PATCH_SIZE = 8
PATCH_STRIDE = 4
N_SITES = 6
POSITIONAL_DIM = 4
STEPS = 90
SEEDS = [0, 1, 2]

TRAIN_IDX = [0, 1]  # cat, ship (hydrofoil)
HELDOUT_IDX = [3, 5, 6]  # airplane, frog, automobile -- unseen classes/images


def cifar_path(idx: int) -> Path:
    return sorted(CIFAR_DIR.glob(f"{idx:04d}_*.png"))[0]


TRAIN_PATHS = [cifar_path(i) for i in TRAIN_IDX]
HELDOUT_PATHS = [cifar_path(i) for i in HELDOUT_IDX]

OUT_DIR = REPO_ROOT / "Main2" / "newHVK" / "results" / "topology_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = OUT_DIR / "real_circuit_confirmation.json"


def load_image_data(path: Path, device: torch.device):
    image = load_grayscale_image(path)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4  # guard zero-norm MPS, same fix used elsewhere in this project
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    return {
        "image": image,
        "raw_positions": raw_positions,
        "features": features_t.to(device),
        "positions": positions.to(device),
        "targets": targets.to(device),
    }


def eval_held_out(model, decoder, device, path: Path) -> dict:
    data = load_image_data(path, device)
    model.eval()
    decoder.eval()
    with torch.no_grad():
        observables, _ = model(data["features"], data["positions"])
        pred = decoder(observables, data["positions"]).cpu().numpy()
    reconstruction = stitch_overlapping_patches(pred, data["raw_positions"], image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    return compute_metrics(reconstruction, data["image"])


def run_topology(topology: str, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device("auto")

    train_data = [load_image_data(p, device) for p in TRAIN_PATHS]
    train_features = torch.cat([d["features"] for d in train_data], dim=0)
    train_positions = torch.cat([d["positions"] for d in train_data], dim=0)
    train_targets = torch.cat([d["targets"] for d in train_data], dim=0)

    if topology == "HVK1D":
        from src.decoder.patch_decoder import PatchDecoder
        from src.quantum.circuit import observable_dim
        from src.quantum.quantum_model import QuantumModel

        model = QuantumModel(feature_dim=train_features.shape[1], positional_dim=POSITIONAL_DIM).to(device)
        decoder = PatchDecoder(observable_dim=observable_dim, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
        lr = 0.003
    elif topology == "HVK2D":
        from Main2.src.model import PatchDecoder as PatchDecoder2D
        from Main2.src.model import Quantum2DGridModel

        model = Quantum2DGridModel(feature_dim=train_features.shape[1], positional_dim=POSITIONAL_DIM).to(device)
        decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
        lr = 0.004
    else:
        raise ValueError(topology)

    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in decoder.parameters())

    t0 = time.perf_counter()
    for step in range(STEPS):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(train_features, train_positions)
        output = decoder(observables, train_positions)
        loss = torch.mean((output - train_targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        if step % 30 == 0 or step == STEPS - 1:
            print(f"  [{topology} seed={seed}] step {step:>3d}: loss={loss.item():.6f}", flush=True)
    elapsed = time.perf_counter() - t0

    held_out_metrics = {p.name: eval_held_out(model, decoder, device, p) for p in HELDOUT_PATHS}
    return {
        "topology": topology, "seed": seed, "n_params": n_params,
        "wall_time_s": elapsed, "steps": STEPS,
        "train_images": [p.name for p in TRAIN_PATHS],
        "held_out_metrics": held_out_metrics,
    }


def main(smoke_test: bool = False):
    global STEPS, SEEDS
    if smoke_test:
        STEPS = 2
        SEEDS = [0]

    print("Train images:", [p.name for p in TRAIN_PATHS])
    print("Held-out images:", [p.name for p in HELDOUT_PATHS])

    results = []
    for topology in ("HVK1D", "HVK2D"):
        for seed in SEEDS:
            print(f"\n=== {topology} seed={seed} ===")
            r = run_topology(topology, seed)
            print(json.dumps(r, indent=2))
            results.append(r)
            RESULT_FILE.write_text(json.dumps({"runs": results}, indent=2, default=str))

    print("\nDone. Saved to", RESULT_FILE)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
