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
from qiskit import QuantumCircuit

from run_ibm_hvk_probe import (
    OUTPUT_DIR,
    angles_from_patch,
    chain_edges,
    counts_from_sampler_result,
    grid_edges,
    order_from_counts,
    run_on_ibm,
)


def parse_epochs(value: str) -> list[int]:
    epochs = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not epochs:
        raise argparse.ArgumentTypeError("Provide at least one epoch, for example 0,5,10.")
    if any(epoch < 0 for epoch in epochs):
        raise argparse.ArgumentTypeError("Epochs must be non-negative.")
    return epochs


def epoch_strength(epoch: int, max_epoch: int) -> float:
    if max_epoch <= 0:
        return 0.05
    progress = epoch / max_epoch
    return float(0.05 + 0.20 * (1.0 - np.exp(-3.0 * progress)))


def build_epoch_circuit(vector: np.ndarray, variant: str, n_qubits: int, epoch: int, max_epoch: int) -> QuantumCircuit:
    angles = angles_from_patch(vector, n_qubits)
    circuit = QuantumCircuit(n_qubits, n_qubits)
    progress = 0.0 if max_epoch <= 0 else epoch / max_epoch
    for qubit in range(n_qubits):
        circuit.ry(float(angles[qubit, 0]) * (0.7 + 0.3 * progress), qubit)
        circuit.rz(float(angles[qubit, 1]) * (0.7 + 0.3 * progress), qubit)

    edges = chain_edges(n_qubits) if variant == "hvk1d" else grid_edges(n_qubits)
    strength = epoch_strength(epoch, max_epoch)
    for source, target in edges:
        circuit.cx(source, target)
        circuit.ry(strength, target)
        circuit.cx(source, target)

    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def save_epoch_plot(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        epochs = sorted({row["epoch"] for row in variant_rows})
        proxy_means = []
        proxy_stds = []
        corr_means = []
        corr_stds = []
        for epoch in epochs:
            epoch_rows = [row for row in variant_rows if row["epoch"] == epoch]
            proxy = np.array([row["hardware_proxy_loss"] for row in epoch_rows], dtype=np.float32)
            corr = np.array([row["mean_zz_correlation"] for row in epoch_rows], dtype=np.float32)
            proxy_means.append(float(proxy.mean()))
            proxy_stds.append(float(proxy.std()))
            corr_means.append(float(corr.mean()))
            corr_stds.append(float(corr.std()))
        axes[0].errorbar(epochs, proxy_means, yerr=proxy_stds, marker="o", capsize=3, label=variant)
        axes[1].errorbar(epochs, corr_means, yerr=corr_stds, marker="o", capsize=3, label=variant)
    axes[0].set_title("Heron proxy loss vs epoch")
    axes[0].set_xlabel("epoch label")
    axes[0].set_ylabel("hardware proxy loss")
    axes[1].set_title("Heron ZZ correlation vs epoch")
    axes[1].set_xlabel("epoch label")
    axes[1].set_ylabel("mean ZZ correlation")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small epoch-labeled HVK hardware proxy sweep on IBM Quantum."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz",
    )
    parser.add_argument("--variant", choices=["hvk1d", "hvk2d", "both"], default="hvk1d")
    parser.add_argument("--backend")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--max-patches", type=int, default=1)
    parser.add_argument("--epochs", type=parse_epochs, default=parse_epochs("0,5,10,25,50"))
    parser.add_argument("--allow-large-heron-job", action="store_true")
    parser.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_qubits < 2:
        raise ValueError("--n-qubits must be at least 2.")
    if args.max_patches < 1:
        raise ValueError("--max-patches must be at least 1.")
    if not args.dry_run and not args.token:
        raise ValueError("No IBM token found. Set IBM_QUANTUM_TOKEN before submitting to hardware.")

    data = np.load(args.dataset, allow_pickle=False)
    vectors = data["patch_vectors"][: args.max_patches]
    variants = ["hvk1d", "hvk2d"] if args.variant == "both" else [args.variant]
    max_epoch = max(args.epochs)

    circuits = []
    labels = []
    for variant in variants:
        for epoch in args.epochs:
            for patch_index, vector in enumerate(vectors):
                circuits.append(build_epoch_circuit(vector, variant, args.n_qubits, epoch, max_epoch))
                labels.append(
                    {
                        "variant": variant,
                        "epoch": epoch,
                        "patch_index": patch_index,
                        "epoch_strength": epoch_strength(epoch, max_epoch),
                    }
                )

    if len(circuits) > 20 and not args.allow_large_heron_job:
        raise ValueError(
            f"This would submit {len(circuits)} circuits to Heron. "
            "Use fewer patches/epochs, or pass --allow-large-heron-job if you really want to spend that quota."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ibm_epoch_probe_circuits.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    **label,
                    "num_qubits": circuit.num_qubits,
                    "depth": circuit.depth(),
                    "num_ops": dict(circuit.count_ops()),
                }
                for label, circuit in zip(labels, circuits)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"Built {len(circuits)} epoch probe circuits. Summary: {summary_path}")
        return

    backend, job_id, result = run_on_ibm(circuits, args.backend, args.shots, args.token, args.n_qubits)
    rows = []
    for index, label in enumerate(labels):
        counts = counts_from_sampler_result(result, index)
        edges = chain_edges(args.n_qubits) if label["variant"] == "hvk1d" else grid_edges(args.n_qubits)
        rows.append({**label, "backend": backend, "job_id": job_id, **order_from_counts(counts, edges, args.n_qubits)})

    output_path = args.output_dir / "ibm_epoch_probe_results.json"
    plot_path = args.output_dir / "ibm_epoch_proxy_loss_vs_epoch.png"
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_epoch_plot(rows, plot_path)
    print(f"IBM epoch probe complete: backend={backend}, job_id={job_id}")
    print(f"Results: {output_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
