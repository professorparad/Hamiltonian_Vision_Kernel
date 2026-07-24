"""MPS bond-dimension sweep of the order-parameter phase-transition diagnostic.

Extends the finite-size study (run_finite_size_phase_transition.py, which
varies quantum qubit count N) along an orthogonal, purely classical axis: the
bond dimension chi of the MPS tensor-network compression used to build HVK's
input features (extract_mps_features(..., bond_dim=chi)). Qubit count is held
fixed at N=6 throughout, so this isolates the effect of classical feature
compression fidelity from quantum circuit width. Same corrected protocol,
same detection rule, same "exploratory_pending_statistical_review" tagging
as the rest of this project's phase-transition work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from run_finite_size_phase_transition import (
    DATASET_LOADERS,
    N_SITES,
    POSITIONAL_DIM,
    PATCH_SIZE,
    PATCH_STRIDE,
    detect_phase_transition,
    order_parameter_from_observables,
    set_seed,
)
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from src.quantum.quantum_model import QuantumModel
from src.decoder.patch_decoder import PatchDecoder
from src.training.training import resolve_device

EPOCHS = 200
LR = 0.004
N_QUBITS = 6  # fixed: isolates bond-dimension effect from qubit-count effect (already studied separately)
BOND_DIMS = [1, 2, 4, 8]
SEEDS = [0, 1]

WORKSPACE = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE / "results" / "bond_dimension_phase_transition"


def train_with_tracking(image: np.ndarray, device: torch.device, epochs: int, seed: int, bond_dim: int) -> dict:
    set_seed(seed)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=bond_dim) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = QuantumModel(feature_dim=features_t.shape[1], positional_dim=POSITIONAL_DIM, qubit_count=N_QUBITS).to(device)
    decoder = PatchDecoder(observable_dim=model.observable_dim, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    order_trace = []
    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_observables, _ = model(features_t, positions)
        order_trace.append(order_parameter_from_observables(eval_observables, N_QUBITS))

    transition = detect_phase_transition(order_trace)
    transition["order_trace"] = order_trace
    return transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dims", nargs="+", type=int, default=BOND_DIMS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--datasets", nargs="+", default=["cifar10"])
    parser.add_argument("--images-per-dataset", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    results_path = OUTPUT_DIR / "bond_dimension_scaling.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else []
    already_done = {(r["bond_dim"], r["dataset"], r["image_index"], r["seed"]) for r in all_results}

    for bond_dim in args.bond_dims:
        for dataset_name in args.datasets:
            images = DATASET_LOADERS[dataset_name](args.images_per_dataset)
            for img_idx, image in enumerate(images):
                for seed in args.seeds:
                    key = (bond_dim, dataset_name, img_idx, seed)
                    if key in already_done:
                        print(f"skip (already done): chi={bond_dim} dataset={dataset_name} image={img_idx} seed={seed}", flush=True)
                        continue
                    print(f"\n=== chi={bond_dim} dataset={dataset_name} image={img_idx} seed={seed} (N={N_QUBITS}) ===", flush=True)
                    try:
                        result = train_with_tracking(image, device, args.epochs, seed, bond_dim)
                    except Exception as exc:
                        print(f"  SKIPPED due to error: {exc}", flush=True)
                        continue
                    record = {
                        "bond_dim": bond_dim,
                        "qubit_count": N_QUBITS,
                        "dataset": dataset_name,
                        "image_index": img_idx,
                        "seed": seed,
                        "trace_mode": "post_update_eval_mode",
                        "evidentiary_status": "exploratory_pending_statistical_review",
                        **result,
                    }
                    all_results.append(record)
                    results_path.write_text(json.dumps(all_results, indent=2))
                    print(
                        f"  critical_epoch={result['critical_epoch']} "
                        f"max_susceptibility={result['max_susceptibility']:.4f} "
                        f"final_order_parameter={result['final_order_parameter']:.4f}",
                        flush=True,
                    )

    summary = {}
    for chi in sorted({r["bond_dim"] for r in all_results}):
        rows = [r for r in all_results if r["bond_dim"] == chi]
        critical_epochs = [r["critical_epoch"] for r in rows if r["detected"]]
        summary[str(chi)] = {
            "n": len(rows),
            "n_detected": len(critical_epochs),
            "mean_critical_epoch": float(np.mean(critical_epochs)) if critical_epochs else None,
            "std_critical_epoch": float(np.std(critical_epochs)) if critical_epochs else None,
        }
    print("\n=== Summary ===")
    for chi, s in summary.items():
        print(f"chi={chi}: n={s['n']} detected={s['n_detected']} mean_t_c={s['mean_critical_epoch']} std={s['std_critical_epoch']}")
    (OUTPUT_DIR / "bond_dimension_scaling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
