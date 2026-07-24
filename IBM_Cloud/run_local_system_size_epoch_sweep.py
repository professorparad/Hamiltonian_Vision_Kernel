from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from run_ibm_epoch_probe import build_epoch_circuit
from run_ibm_hvk_probe import chain_edges, grid_edges, order_from_counts

DATASETS = {
    "monalisa": Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz",
    "cifar": Path(__file__).resolve().parent / "datasets" / "cifar_patches.npz",
}
N_QUBITS_LIST = [4, 6, 8]
EPOCHS = [0, 5, 10, 25, 50]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "system_size_epoch_sweep"


def run_dataset_sweep(dataset_path: Path, variant: str, shots: int, patch_index: int) -> list[dict]:
    data = np.load(dataset_path, allow_pickle=False)
    vector = data["patch_vectors"][patch_index]
    max_epoch = max(EPOCHS)
    backend = AerSimulator()

    rows = []
    for n_qubits in N_QUBITS_LIST:
        edges = chain_edges(n_qubits) if variant == "hvk1d" else grid_edges(n_qubits)
        circuits = [build_epoch_circuit(vector, variant, n_qubits, epoch, max_epoch) for epoch in EPOCHS]
        transpiled = transpile(circuits, backend=backend, optimization_level=1)
        result = backend.run(transpiled, shots=shots).result()
        for epoch, circuit in zip(EPOCHS, transpiled):
            counts = result.get_counts(circuit)
            metrics = order_from_counts(counts, edges, n_qubits)
            rows.append({"n_qubits": n_qubits, "epoch": epoch, **metrics})
    return rows


def save_plot(rows: list[dict], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for n_qubits in N_QUBITS_LIST:
        n_rows = sorted((row for row in rows if row["n_qubits"] == n_qubits), key=lambda row: row["epoch"])
        ax.plot(
            [row["epoch"] for row in n_rows],
            [row["mean_order_parameter"] for row in n_rows],
            marker="o",
            label=f"N = {n_qubits}",
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Order parameter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Aer-simulator baseline: order_parameter vs epoch for N=4/6/8, no IBM hardware quota used."
    )
    parser.add_argument("--variant", choices=["hvk1d", "hvk2d"], default="hvk1d")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_path in DATASETS.items():
        rows = run_dataset_sweep(dataset_path, args.variant, args.shots, args.patch_index)
        json_path = args.output_dir / f"{dataset_name}_local_order_parameter_vs_epoch.json"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        plot_path = args.output_dir / f"{dataset_name}_local_order_parameter_vs_epoch.png"
        save_plot(rows, f"Order parameter vs epoch ({dataset_name}, local simulator)", plot_path)
        print(f"{dataset_name}: results={json_path} plot={plot_path}")


if __name__ == "__main__":
    main()
