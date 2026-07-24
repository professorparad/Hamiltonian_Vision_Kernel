"""Change-magnitude view of the energy-to-entanglement-ratio diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "critical_temperature" / "critical_temperature_cifar10.json"
OUTPUT_PATHS = [
    ROOT / "latex_outputs" / "paper_latex" / "figures" / "critical_temperature_susceptibility.pdf",
    ROOT / "Main2" / "newHVK" / "results" / "critical_temperature" / "critical_temperature_susceptibility.png",
]


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    rows = sorted(rows, key=lambda r: (r["image_index"], r["seed"]))

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for ax, row in zip(axes.flat, rows):
        trace = row["t_eff_trace"]
        diffs = [0.0] + [abs(trace[i] - trace[i - 1]) for i in range(1, len(trace))]
        diffs = np.array(diffs)
        epochs = np.arange(len(diffs))
        color = "tab:blue" if row["image_index"] == 0 else "tab:red"
        ax.plot(epochs, diffs, color=color, linewidth=1.1)
        ax.axhline(row["threshold"], color="black", linestyle=":", linewidth=1, alpha=0.7, label="threshold")
        if row["detected"]:
            tc = row["critical_epoch"]
            ax.axvline(tc, color="black", linestyle="--", alpha=0.6, linewidth=1.1)
            ax.set_title(f"image={row['image_index']} seed={row['seed']}  DETECTED @ t={tc}", fontsize=9, color="darkgreen")
        else:
            ax.set_title(f"image={row['image_index']} seed={row['seed']}  not detected", fontsize=9, color="gray")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|\Delta R_{ES}|$")

    fig.suptitle(
        "Descriptive change detection in the energy-to-entanglement ratio\n"
        "(dotted line: within-run threshold; dashed line: detected maximum)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
