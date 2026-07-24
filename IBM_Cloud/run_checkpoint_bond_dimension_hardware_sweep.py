"""Real-hardware replay of the MPS bond-dimension comparison.

Same checkpoint-replay methodology as run_checkpoint_hardware_sweep.py
(order parameter, N swept) and run_checkpoint_temperature_hardware_sweep.py
(R_ES, N=6 fixed), applied to the remaining untested axis: MPS bond
dimension chi, with qubit count fixed at N=6 (matching
run_bond_dimension_phase_transition.py's simulated sweep). Trains real
checkpoints at chi in {1,2,4,8}, saves learned parameters at eight epoch
labels, rebuilds each checkpoint's circuit gate-for-gate, and executes on
real hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = ROOT / "Main"
BENCH_DIR = ROOT / "Baselines" / "cifar10_comparisons"
for p in (MAIN_DIR, BENCH_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from src.quantum.quantum_model import QuantumModel
from src.decoder.patch_decoder import PatchDecoder

from run_ibm_hvk_probe import chain_edges, counts_from_sampler_result, order_from_counts, run_on_ibm
from run_cross_quantum_validation import extract_qiskit_counts, run_on_ionq
from run_checkpoint_hardware_sweep import DATASETS, load_image, state_prep_gates

PATCH_SIZE = 8
PATCH_STRIDE = 8
N_SITES = 6
N_QUBITS = 6  # fixed: isolates bond-dimension effect, matches run_bond_dimension_phase_transition.py
EPOCHS = 200
LR = 0.004
EPOCH_LABELS = [0, 5, 10, 25, 50, 100, 150, 200]
BOND_DIMS = [1, 2, 4, 8]
SHOTS_LIST = [256, 512, 1024]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "checkpoint_bond_dimension_hardware_sweep"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_checkpoint(image: np.ndarray, bond_dim: int, seed: int, epochs: int) -> dict[int, dict]:
    set_seed(seed)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=bond_dim) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=4)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    model = QuantumModel(feature_dim=features_t.shape[1], positional_dim=4, qubit_count=N_QUBITS)
    decoder = PatchDecoder(observable_dim=model.observable_dim, positional_dim=4, patch_size=PATCH_SIZE)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    checkpoints: dict[int, dict] = {}

    def save_checkpoint(epoch_label: int) -> None:
        model.eval()
        with torch.no_grad():
            proj_features = model.feature_projection(features_t).numpy()
            proj_positions = model.position_projection(positions).numpy()
        checkpoints[epoch_label] = {
            "weights": model.weights.detach().numpy().copy(),
            "proj_features": proj_features.copy(),
            "proj_positions": proj_positions.copy(),
        }

    if 0 in EPOCH_LABELS:
        save_checkpoint(0)
    for step in range(1, epochs + 1):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        if step in EPOCH_LABELS:
            save_checkpoint(step)

    return checkpoints


def build_circuit(checkpoint: dict, patch_index: int = 0) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    inputs = checkpoint["proj_features"][patch_index]
    positional_angles = checkpoint["proj_positions"][patch_index]
    weights = checkpoint["weights"]
    state_prep_gates(qc, inputs, positional_angles, weights, N_QUBITS)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def build_all_circuits(all_checkpoints: dict) -> tuple[list[QuantumCircuit], list[dict]]:
    circuits = []
    labels = []
    for dataset_name, per_chi in all_checkpoints.items():
        for bond_dim, checkpoints in per_chi.items():
            for epoch_label, checkpoint in checkpoints.items():
                circuits.append(build_circuit(checkpoint))
                labels.append({"dataset": dataset_name, "bond_dim": bond_dim, "epoch": epoch_label})
    return circuits, labels


def save_plot(rows: list[dict], dataset_name: str, shots: int, provider: str, backend: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for bond_dim in BOND_DIMS:
        n_rows = sorted(
            (row for row in rows if row["dataset"] == dataset_name and row["bond_dim"] == bond_dim),
            key=lambda row: row["epoch"],
        )
        if not n_rows:
            continue
        ax.plot(
            [row["epoch"] for row in n_rows],
            [row["mean_order_parameter"] for row in n_rows],
            marker="o",
            label=f"chi = {bond_dim}",
        )
    ax.set_title(f"Order parameter vs epoch (REAL trained checkpoints)\n{dataset_name}, {provider} {backend}, shots={shots}, N=6")
    ax.set_xlabel("Epoch (real gradient-descent steps)")
    ax.set_ylabel("Order parameter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--providers", nargs="+", choices=["ibm", "ionq"], default=["ibm"])
    parser.add_argument("--ibm-backend", default="ibm_marrakesh")
    parser.add_argument("--ionq-backend", default="ionq_simulator")
    parser.add_argument("--shots", type=int, action="append", choices=SHOTS_LIST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots_list = args.shots or SHOTS_LIST
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Training real N=6 checkpoints across bond dimensions (Mona Lisa + CIFAR)...", flush=True)
    all_checkpoints: dict[str, dict[int, dict[int, dict]]] = {}
    for dataset_name, image_path in DATASETS.items():
        image = load_image(image_path)
        all_checkpoints[dataset_name] = {}
        for bond_dim in BOND_DIMS:
            print(f"  training {dataset_name} chi={bond_dim}...", flush=True)
            all_checkpoints[dataset_name][bond_dim] = train_and_checkpoint(image, bond_dim, args.seed, args.epochs)

    circuits, labels = build_all_circuits(all_checkpoints)
    print(f"Built {len(circuits)} circuits ({len(DATASETS)} datasets x {len(BOND_DIMS)} bond dims x {len(EPOCH_LABELS)} epoch checkpoints).", flush=True)

    if args.dry_run:
        for provider in args.providers:
            for shots in shots_list:
                print(f"[dry-run] would submit {len(circuits)} circuits to {provider} at shots={shots}")
        return

    edges = chain_edges(N_QUBITS)

    for provider in args.providers:
        backend_name = args.ibm_backend if provider == "ibm" else args.ionq_backend
        for shots in shots_list:
            if provider == "ibm":
                backend, job_id, result = run_on_ibm(circuits, backend_name, shots, os.environ.get("IBM_QUANTUM_TOKEN"), N_QUBITS)
            else:
                backend, job_id, result = run_on_ionq(circuits, backend_name, shots)

            rows = []
            for index, label in enumerate(labels):
                counts = counts_from_sampler_result(result, index) if provider == "ibm" else extract_qiskit_counts(result, index)
                rows.append({**label, "backend": backend, "job_id": job_id, **order_from_counts(counts, edges, N_QUBITS)})

            json_path = args.output_dir / f"{provider}_{backend}_shots{shots}.json"
            json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"provider={provider} shots={shots}: backend={backend} job_id={job_id} results={json_path}")

            for dataset_name in DATASETS:
                plot_path = args.output_dir / f"{dataset_name}_{provider}_{backend}_shots{shots}_order_parameter_vs_epoch.png"
                save_plot(rows, dataset_name, shots, provider, backend, plot_path)
                print(f"  plot: {plot_path}")


if __name__ == "__main__":
    main()
