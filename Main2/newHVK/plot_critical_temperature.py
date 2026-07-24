"""Energy-to-entanglement-ratio trace figure for paper_hvk.tex."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "critical_temperature" / "critical_temperature_cifar10.json"
OUTPUT_PATHS = [
    ROOT / "latex_outputs" / "paper_latex" / "figures" / "critical_temperature_traces.pdf",
    ROOT / "Main2" / "newHVK" / "results" / "critical_temperature" / "critical_temperature_traces.png",
]

SHADES = {0: ["#1f4e8b", "#4a7fc1", "#7fabde", "#b3cfec"], 1: ["#8b1f1f", "#c14a4a", "#de7f7f", "#ecb3b3"]}


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, ax = plt.subplots(figsize=(7.5, 5))

    run_index = {0: 0, 1: 0}
    for row in sorted(rows, key=lambda r: (r["image_index"], r["seed"])):
        img = row["image_index"]
        trace = row["t_eff_trace"]
        epochs = list(range(len(trace)))
        color = SHADES[img][run_index[img] % len(SHADES[img])]
        run_index[img] += 1
        label = f"{row['image_name'].split('_')[1] if '_' in row['image_name'] else row['image_name']} seed={row['seed']}"
        ax.plot(epochs, trace, color=color, alpha=0.8, linewidth=1.3, label=label)
        if row["detected"]:
            tc = row["critical_epoch"]
            ax.axvline(tc, color=color, linestyle="--", alpha=0.5, linewidth=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Signed energy-to-entanglement ratio $R_{ES}(t)=H(t)/S$")
    ax.set_title("Energy-to-entanglement ratio vs epoch, HVK1D\n"
                 "two CIFAR-10 images x four seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
