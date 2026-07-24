"""Paper-ready summary figure: R_ES(t) = H(t)/S from REAL trained checkpoints
replayed on real IBM hardware, across shots=256/512/1024, Mona Lisa + CIFAR."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "checkpoint_temperature_hardware_sweep"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "latex_outputs" / "paper_latex" / "figures"
OUTPUT_PATHS = [
    FIGURES_DIR / "checkpoint_hardware_energy_entanglement_ratio.pdf",
    RESULTS_DIR / "checkpoint_hardware_energy_entanglement_ratio.png",
]
DATASETS = ["monalisa", "cifar"]
SHOTS_LIST = [256, 512, 1024]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dataset_name in zip(axes, DATASETS):
        for shots in SHOTS_LIST:
            rows = json.loads((RESULTS_DIR / f"ibm_ibm_marrakesh_shots{shots}.json").read_text())
            d_rows = sorted((r for r in rows if r["dataset"] == dataset_name), key=lambda r: r["epoch"])
            ax.plot([r["epoch"] for r in d_rows], [r["r_es"] for r in d_rows], marker="o", label=f"shots={shots}")
        ax.set_title(dataset_name)
        ax.set_xlabel("Epoch (real gradient-descent steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$R_{ES}(t) = H(t)/S$ (real hardware)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Energy-to-entanglement ratio vs epoch, real trained checkpoints on ibm_marrakesh")
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
