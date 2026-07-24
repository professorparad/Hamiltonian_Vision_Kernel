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

from run_ibm_epoch_probe import build_epoch_circuit
from run_ibm_hvk_probe import chain_edges, grid_edges, order_from_counts
from run_cross_quantum_validation import extract_qiskit_counts, run_on_ionq

DATASETS = {
    "monalisa": Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz",
    "cifar": Path(__file__).resolve().parent / "datasets" / "cifar_patches.npz",
}
N_QUBITS_LIST = [4, 6, 8]
EPOCHS = [0, 5, 10, 25, 50]
SHOTS_LIST = [256, 512, 1024]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "system_size_epoch_sweep"


def build_circuits(variant: str, patch_index: int):
    circuits = []
    labels = []
    max_epoch = max(EPOCHS)
    for dataset_name, dataset_path in DATASETS.items():
        data = np.load(dataset_path, allow_pickle=False)
        vector = data["patch_vectors"][patch_index]
        for n_qubits in N_QUBITS_LIST:
            for epoch in EPOCHS:
                circuits.append(build_epoch_circuit(vector, variant, n_qubits, epoch, max_epoch))
                labels.append({"dataset": dataset_name, "n_qubits": n_qubits, "epoch": epoch})
    return circuits, labels


def save_plot(rows: list[dict], dataset_name: str, shots: int, backend: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for n_qubits in N_QUBITS_LIST:
        n_rows = sorted(
            (row for row in rows if row["dataset"] == dataset_name and row["n_qubits"] == n_qubits),
            key=lambda row: row["epoch"],
        )
        if not n_rows:
            continue
        ax.plot(
            [row["epoch"] for row in n_rows],
            [row["mean_order_parameter"] for row in n_rows],
            marker="o",
            label=f"N = {n_qubits}",
        )
    ax.set_title(f"Order parameter vs epoch ({dataset_name}, IonQ {backend}, shots={shots})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Order parameter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit epoch-labeled HVK circuits across n_qubits=4/6/8 to IonQ for each shots budget."
    )
    parser.add_argument("--variant", choices=["hvk1d", "hvk2d"], default="hvk1d")
    parser.add_argument("--backend", default="ionq_simulator", choices=["ionq_simulator", "ionq_qpu"])
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--shots", type=int, action="append", choices=SHOTS_LIST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots_list = args.shots or SHOTS_LIST
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circuits, labels = build_circuits(args.variant, args.patch_index)
    print(f"Built {len(circuits)} circuits per shots job ({len(DATASETS)} datasets x {len(N_QUBITS_LIST)} qubit counts x {len(EPOCHS)} epochs).")

    if args.dry_run:
        for shots in shots_list:
            print(f"[dry-run] would submit {len(circuits)} circuits to {args.backend} at shots={shots}")
        return

    for shots in shots_list:
        backend, job_id, result = run_on_ionq(circuits, args.backend, shots)
        rows = []
        for index, label in enumerate(labels):
            counts = extract_qiskit_counts(result, index)
            n_qubits = label["n_qubits"]
            edges = chain_edges(n_qubits) if args.variant == "hvk1d" else grid_edges(n_qubits)
            rows.append({**label, "backend": backend, "job_id": job_id, **order_from_counts(counts, edges, n_qubits)})

        json_path = args.output_dir / f"ionq_{args.backend}_shots{shots}.json"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"shots={shots}: backend={backend} job_id={job_id} results={json_path}")

        for dataset_name in DATASETS:
            plot_path = args.output_dir / f"{dataset_name}_ionq_{args.backend}_shots{shots}_order_parameter_vs_epoch.png"
            save_plot(rows, dataset_name, shots, backend, plot_path)
            print(f"  plot: {plot_path}")


if __name__ == "__main__":
    main()
