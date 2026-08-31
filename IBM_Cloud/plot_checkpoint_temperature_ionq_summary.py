"""Paper-ready summary figure: R_ES(t) = H(t)/S from REAL trained checkpoints
replayed on the IonQ ideal cloud simulator (1024 shots, the only shot budget
submitted for this diagnostic), Mona Lisa + CIFAR. Companion to
plot_checkpoint_temperature_hardware_summary.py's IBM figure, using the IonQ
JSON already archived by run_checkpoint_temperature_hardware_sweep.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "checkpoint_temperature_hardware_sweep"
RESULTS_PATH = RESULTS_DIR / "ionq_ionq_simulator_shots1024.json"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "overleaf_docs" / "assets" / "figures"
OUTPUT_PATHS = [
    FIGURES_DIR / "checkpoint_ionq_energy_entanglement_ratio.pdf",
    RESULTS_DIR / "checkpoint_ionq_energy_entanglement_ratio.png",
]
DATASETS = ["monalisa", "cifar"]


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dataset_name in zip(axes, DATASETS):
        d_rows = sorted((r for r in rows if r["dataset"] == dataset_name), key=lambda r: r["epoch"])
        ax.plot(
            [r["epoch"] for r in d_rows],
            [r["r_es"] for r in d_rows],
            marker="o",
            color="tab:purple",
            label="shots=1024",
        )
        ax.set_title(dataset_name)
        ax.set_xlabel("Epoch (real gradient-descent steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$R_{ES}(t) = H(t)/S$ (IonQ simulator)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Energy-to-entanglement ratio vs epoch, real trained checkpoints on ionq_simulator")
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
