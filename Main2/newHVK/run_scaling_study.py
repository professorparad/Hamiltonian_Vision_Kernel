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
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)

    grad_stats = gradient_variance(model, decoder, data["features"], data["positions"], data["targets"], seed=seed)
    # restore a fresh training-init weight tensor after the gradient probes perturbed it
    if n_layers is not None:
        set_depth(model, topology, n_layers, qubit_count, device, seed)

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


def main(smoke_test: bool = False):
    global STEPS, SEEDS
    if smoke_test:
        STEPS = 2
        SEEDS = [0]

    results = {"qubit_sweep": [], "bond_dim_sweep": [], "depth_sweep": []}

    print("=== Qubit-count sweep (HVK1D only) ===")
    for q in ([4, 6, 8] if not smoke_test else [4]):
        for seed in SEEDS:
            r = run_training("HVK1D", seed, qubit_count=q, bond_dim=4, steps=STEPS)
            print(json.dumps(r))
            results["qubit_sweep"].append(r)
            RESULT_FILE.write_text(json.dumps(results, indent=2, default=str))

    print("\n=== Bond-dimension sweep (both topologies) ===")
    for topology in ("HVK1D", "HVK2D"):
        for chi in ([1, 2, 4, 8] if not smoke_test else [1]):
            for seed in SEEDS:
                r = run_training(topology, seed, qubit_count=6, bond_dim=chi, steps=STEPS)
                print(json.dumps(r))
                results["bond_dim_sweep"].append(r)
                RESULT_FILE.write_text(json.dumps(results, indent=2, default=str))

    print("\n=== Circuit-depth sweep (both topologies) ===")
    for topology in ("HVK1D", "HVK2D"):
        for depth in ([1, 2, 3, 4] if not smoke_test else [1]):
            for seed in SEEDS:
                r = run_training(topology, seed, qubit_count=6, bond_dim=4, n_layers=depth, steps=STEPS)
                print(json.dumps(r))
                results["depth_sweep"].append(r)
                RESULT_FILE.write_text(json.dumps(results, indent=2, default=str))

    print("\nDone. Saved to", RESULT_FILE)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
