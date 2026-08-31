"""Paper-ready summary figure: order parameter vs epoch from REAL trained
checkpoints replayed on the IonQ ideal cloud simulator, Mona Lisa + CIFAR
side by side. Companion to plot_checkpoint_hardware_summary.py's IBM figure,
using the IonQ JSON already archived by run_checkpoint_hardware_sweep.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP_DIR = Path(__file__).resolve().parent / "outputs" / "checkpoint_hardware_sweep"
RESULTS_PATH = SWEEP_DIR / "ionq_ionq_simulator_shots1024.json"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "overleaf_docs" / "assets" / "figures"
OUTPUT_PATHS = [
    FIGURES_DIR / "checkpoint_ionq_order_parameter.pdf",
    SWEEP_DIR / "checkpoint_ionq_order_parameter.png",
]
DATASETS = ["monalisa", "cifar"]
N_QUBITS_LIST = [4, 6, 8]


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dataset_name in zip(axes, DATASETS):
        for n_qubits in N_QUBITS_LIST:
            n_rows = sorted(
                (r for r in rows if r["dataset"] == dataset_name and r["n_qubits"] == n_qubits),
                key=lambda r: r["epoch"],
            )
            ax.plot(
                [r["epoch"] for r in n_rows],
                [r["mean_order_parameter"] for r in n_rows],
                marker="o",
                label=f"N={n_qubits}",
            )
        ax.set_title(dataset_name)
        ax.set_xlabel("Epoch (real gradient-descent steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"Order parameter $M_z$ (IonQ simulator)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Order parameter vs epoch, real trained checkpoints\n"
        "replayed on ionq_simulator (1024 shots)"
    )
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
