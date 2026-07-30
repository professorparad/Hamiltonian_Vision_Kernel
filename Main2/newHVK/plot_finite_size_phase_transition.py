"""System-size sensitivity figure for the order-parameter change diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "finite_size_phase_transition" / "finite_size_scaling.json"
OUTPUT_PATHS = [
    ROOT / "latex_outputs" / "paper_latex" / "figures" / "finite_size_phase_transition.pdf",
    ROOT / "Main2" / "newHVK" / "results" / "finite_size_phase_transition" / "finite_size_phase_transition.png",
]

COLORS = {4: "tab:blue", 6: "tab:red", 8: "tab:green"}
SHADES = {
    4: ["#1f77b4", "#5aa2d4", "#0d4f8b", "#8fc1e6"],
    6: ["#d62728", "#e8696a", "#a3181a", "#f0a3a3"],
    8: ["#2ca02c", "#6fc26f", "#146414", "#a8dda8"],
}
# Distinct linestyle per run-within-N, on top of the color/shade coding above,
# so individual runs stay distinguishable in grayscale or for colorblind readers.
RUN_LINESTYLES = ["-", "--", ":", "-."]


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, ax = plt.subplots(figsize=(8, 5.2))

    run_index = {4: 0, 6: 0, 8: 0}
    for row in sorted(rows, key=lambda r: (r["qubit_count"], r["dataset"], r["seed"])):
        n = row["qubit_count"]
        trace = row["order_trace"]
        epochs = list(range(len(trace)))
        idx = run_index[n]
        color = SHADES[n][idx % len(SHADES[n])]
        linestyle = RUN_LINESTYLES[idx % len(RUN_LINESTYLES)]
        run_index[n] += 1
        label = f"N={n} ({row['dataset']}, seed={row['seed']})"
        ax.plot(epochs, trace, color=color, linestyle=linestyle, alpha=0.8, linewidth=1.3, label=label)
        if row["detected"]:
            tc = row["critical_epoch"]
            ax.axvline(tc, color=color, linestyle=":", alpha=0.5, linewidth=0.9)

    # Summary markers for the mean detected change epoch per N
    for n in (4, 6, 8):
        detected = [r["critical_epoch"] for r in rows if r["qubit_count"] == n and r["detected"]]
        if detected:
            mean_tc = sum(detected) / len(detected)
            ax.axvline(mean_tc, color=COLORS[n], linestyle="-", linewidth=2.2, alpha=0.9,
                       label=f"N={n} mean $t_c$={mean_tc:.0f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Order parameter $M_z$")
    ax.set_title("System-size sensitivity: order parameter vs epoch (N=4, 6, 8)\n"
                 "dashed lines mark each run's detected change epoch")
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
