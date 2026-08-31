"""Paper-ready summary figure: R_ES vs epoch from REAL trained checkpoints
across MPS bond dimension (N=6 fixed), replayed on real IBM hardware."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP_DIR = Path(__file__).resolve().parent / "outputs" / "checkpoint_bond_dimension_temperature_hardware_sweep"
RESULTS_PATH = SWEEP_DIR / "ibm_ibm_marrakesh_shots1024.json"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "overleaf_docs" / "assets" / "figures"
OUTPUT_PATHS = [
    FIGURES_DIR / "checkpoint_hardware_bond_dimension_r_es.pdf",
    SWEEP_DIR / "checkpoint_hardware_bond_dimension_r_es.png",
]
DATASETS = ["monalisa", "cifar"]
# chi=1 excluded: bond dimension 1 means an unentangled MPS product state, so S_total
# collapses to the numerical floor (~5e-6, vs ~0.01-0.05 at chi>=2) and R_ES=H/S diverges
# as a division-by-near-zero artifact, not a physical signal.
BOND_DIMS = [2, 4, 8]


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dataset_name in zip(axes, DATASETS):
        for bond_dim in BOND_DIMS:
            d_rows = sorted(
                (r for r in rows if r["dataset"] == dataset_name and r["bond_dim"] == bond_dim),
                key=lambda r: r["epoch"],
            )
            ax.plot(
                [r["epoch"] for r in d_rows],
                [r["r_es"] for r in d_rows],
                marker="o",
                label=f"chi={bond_dim}",
            )
        ax.set_title(dataset_name)
        ax.set_xlabel("Epoch (real gradient-descent steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$R_{ES}(t) = H(t)/S$ (real hardware)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "R_ES vs epoch across MPS bond dimension,\n"
        "real trained checkpoints on ibm_marrakesh (1024 shots)"
    )
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
