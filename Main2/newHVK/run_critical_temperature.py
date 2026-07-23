"""Thermodynamically-motivated critical-temperature diagnostic, complementing
the order-parameter phase-transition result of run_phase_transition_multi_dataset.py.

Local bond temperature: T_i = <h_i> / (k_B * S_i), where h_i is HVK1D's own
learned Heisenberg bond energy h_i = Jx_i<X_iX_i+1> + Jy_i<Y_iY_i+1> +
Jz_i<Z_iZ_i+1> (already the model's existing energy diagnostic), and S_i is
the STATIC von Neumann bipartite entanglement entropy of that bond from the
classical MPS decomposition of the input patch (already computed as part of
HVK's feature extraction, unrelated to training). k_B = 1 (natural units,
consistent with every other HVK energy term in this project).

This only has a clean bond-for-bond correspondence for HVK1D at n_sites=6
(6-qubit chain, 5 bonds, full Jx/Jy/Jz couplings) -- the classical MPS chain
bonds and the quantum circuit bonds are then literally the same 5 bonds.
HVK2D's grid model only has a single ZZ-type coupling per edge, so it does
not fit this construction as cleanly and is not used here.

Effective temperature T_eff(t) = H(t) / S, with H(t) = sum_i h_i(t) (the
model's own tracked total bond energy) and S = sum_i S_i (fixed). Tracked
across training in noise-free evaluation mode (mirrors the corrected
phase-transition protocol exactly), with the same median-plus-two-std
critical-epoch detection rule applied to |dT_eff/dt|.
"""
from __future__ import annotations

import json
import sys
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

from common import load_grayscale_image
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from src.quantum.quantum_model import QuantumModel
from src.decoder.patch_decoder import PatchDecoder
from src.quantum.circuit import observable_dim
from src.training.training import resolve_device

CIFAR_DIR = BENCH_ROOT / "datasets" / "images"
PATCH_SIZE = 8
PATCH_STRIDE = 8  # non-overlapping, matching run_phase_transition_multi_dataset.py exactly
N_SITES = 6  # == N_QUBITS, so classical MPS bonds == quantum circuit bonds, 1:1
N_QUBITS = 6
N_BONDS = N_QUBITS - 1
POSITIONAL_DIM = 4
IMAGE_SIZE = 32
EPOCHS = 200
LR = 0.003

OUT_DIR = REPO_ROOT / "Main2" / "newHVK" / "results" / "critical_temperature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature layout for n_sites=6 (22-D): 12 local (Z,X x 6 sites) + 5 ZZ + 5 entropy
ENTROPY_SLICE = slice(17, 22)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_critical_point(trace: list[float]) -> dict:
    diffs = [0.0] + [abs(trace[i] - trace[i - 1]) for i in range(1, len(trace))]
    diffs_arr = np.array(diffs)
    threshold = float(np.median(diffs_arr) + 2 * np.std(diffs_arr))
    idx = int(np.argmax(diffs_arr))
    peak = float(diffs_arr[idx])
    detected = bool(peak > threshold and peak > 0)
    return {"critical_epoch": idx if detected else -1, "peak_rate_of_change": peak, "threshold": threshold, "detected": detected, "final_value": trace[-1]}


def train_with_temperature_tracking(image: np.ndarray, device: torch.device, epochs: int, seed: int) -> dict:
    set_seed(seed)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4

    features_list = [extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in safe_patches]
    features = np.array(features_list)
    # static per-bond entropies S_i: mean over patches (S_i is a property of
    # each classical patch's MPS; the model sees all patches jointly, so we
    # use the per-image mean bond entropy as the fixed S_i for that image)
    bond_entropies = features[:, ENTROPY_SLICE].mean(axis=0)  # shape (5,)
    bond_entropies = np.maximum(bond_entropies, 1e-6)  # guard against S_i ~ 0

    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = QuantumModel(feature_dim=features_t.shape[1], positional_dim=POSITIONAL_DIM, qubit_count=N_QUBITS).to(device)
    decoder = PatchDecoder(observable_dim=observable_dim, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    S_i = torch.tensor(bond_entropies, dtype=torch.float32, device=device)
    S_total = float(S_i.sum().item())

    t_eff_trace, h_total_trace = [], []
    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()

        # noise-free evaluation-mode pass for the temperature diagnostic itself
        model.eval()
        with torch.no_grad():
            eval_obs, _ = model(features_t, positions)
            # observable layout: 6 Z, 6 X, 5 ZZ, 5 XX, 5 YY
            zz = eval_obs[:, 12:17].mean(dim=0)  # mean over patches, per bond
            xx = eval_obs[:, 17:22].mean(dim=0)
            yy = eval_obs[:, 22:27].mean(dim=0)
            h_i = model.Jz.detach() * zz + model.Jx.detach() * xx + model.Jy.detach() * yy  # (5,)
            H_total = float(h_i.sum().item())
        t_eff = H_total / S_total
        t_eff_trace.append(t_eff)
        h_total_trace.append(H_total)

    result = detect_critical_point(t_eff_trace)
    result["t_eff_trace"] = t_eff_trace
    result["h_total_trace"] = h_total_trace
    result["bond_entropies"] = bond_entropies.tolist()
    result["S_total"] = S_total
    return result


def main():
    device = resolve_device("auto")
    print(f"Using device: {device}", flush=True)

    image_paths = sorted(CIFAR_DIR.glob("*.png"))[:2]  # cat, ship (hydrofoil) -- same 2 images as the CIFAR-10 phase-transition rerun
    seeds = [0, 1]

    results = []
    for idx, path in enumerate(image_paths):
        image = load_grayscale_image(path)
        for seed in seeds:
            print(f"\n=== image {idx} ({path.name}), seed {seed} ===", flush=True)
            r = train_with_temperature_tracking(image, device, EPOCHS, seed)
            print(f"  critical_epoch={r['critical_epoch']} peak_rate={r['peak_rate_of_change']:.4f} "
                  f"final_T_eff={r['final_value']:.4f} S_total={r['S_total']:.3f}", flush=True)
            r["image_index"] = idx
            r["image_name"] = path.name
            r["seed"] = seed
            results.append(r)
            (OUT_DIR / "critical_temperature_cifar10.json").write_text(json.dumps(results, indent=2))

    detected = [r for r in results if r["detected"]]
    print(f"\n=== Summary: {len(detected)}/{len(results)} detected ===")
    print("Done. Saved to", OUT_DIR / "critical_temperature_cifar10.json")


if __name__ == "__main__":
    main()
