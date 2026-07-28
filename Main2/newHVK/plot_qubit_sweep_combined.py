"""Combined single-figure view of the qubit-count sweep: order parameter vs
epoch for N in {2, 4, 6, 8}, overlaid in one image per dataset, in the same
style as the paper's existing finite-size figures (mean +/- std across seeds,
shaded; dashed vertical line marks each N's detected critical epoch) -- so the
critical-epoch shift across qubit count is visible at a glance, instead of
across four separate per-N plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "qubit_energy_phase_transition" / "qubit_energy_scaling.json"
OUTPUT_DIR = ROOT / "Main2" / "newHVK" / "results" / "qubit_energy_phase_transition"
FIGURES_DIR = ROOT / "latex_outputs" / "paper_latex" / "figures"

QUBIT_COUNTS = [2, 4, 6, 8]
COLORS = {2: "tab:blue", 4: "tab:green", 6: "tab:red", 8: "tab:purple"}


def plot_combined(image_name: str, rows: list[dict], output_paths: list[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for qc in QUBIT_COUNTS:
        runs = [r for r in rows if r["image"] == image_name and r["qubit_count"] == qc]
        if not runs:
            continue
        traces = np.array([r["order_trace"] for r in runs])
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0)
        epochs = np.arange(traces.shape[1])
        color = COLORS[qc]

        critical_epochs = [r["order_transition"]["critical_epoch"] for r in runs if r["order_transition"]["detected"]]
        label = f"N={qc}"
        if critical_epochs:
            label += f" (t_c={np.mean(critical_epochs):.0f})"

        axes[0].plot(epochs, mean_trace, color=color, label=label, linewidth=2)
        axes[0].fill_between(epochs, mean_trace - std_trace, mean_trace + std_trace, color=color, alpha=0.15)
        for tc in critical_epochs:
            axes[0].axvline(tc, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

        susceptibility = np.array(
            [[0.0] + [abs(t[i] - t[i - 1]) for i in range(1, len(t))] for t in traces]
        ).mean(axis=0)
        axes[1].plot(epochs, susceptibility, color=color, label=label, linewidth=2)
        for tc in critical_epochs:
            axes[1].axvline(tc, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Order parameter $M_z(t)$")
    axes[0].set_title(f"Order parameter vs epoch across N, {image_name}\n(mean +/- std over seeds; dashed = detected critical epoch)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Susceptibility $|\\Delta M_z(t)|$")
    axes[1].set_title("Phase transition signal across N")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Descriptive training-dynamics change-point diagnostic (not a physical phase transition)", fontsize=10, y=1.02)
    fig.tight_layout()
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    images = sorted({r["image"] for r in rows})
    for image_name in images:
        combos = {(r["qubit_count"], r["seed"]) for r in rows if r["image"] == image_name}
        have_n = {qc for qc, _ in combos}
        if not set(QUBIT_COUNTS).issubset(have_n):
            print(f"skip {image_name}: only have N={sorted(have_n)}, need {QUBIT_COUNTS}")
            continue
        name_map = {
            "monalisa": "monalisa",
            "0000_cat_domestic_cat_s_000907": "cat",
            "0001_ship_hydrofoil_s_000078": "ship_hydrofoil",
            "0002_ship_sea_boat_s_001456": "ship_seaboat",
            "0003_airplane_jetliner_s_001705": "airplane",
        }
        safe_name = name_map.get(image_name, image_name.split("_")[0])
        output_paths = [
            OUTPUT_DIR / f"{image_name}_qubit_sweep_combined.png",
            FIGURES_DIR / f"qubit_sweep_combined_{safe_name}.pdf",
        ]
        plot_combined(image_name, rows, output_paths)


if __name__ == "__main__":
    main()
