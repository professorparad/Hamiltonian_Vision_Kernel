"""Order-parameter-vs-epoch figure for the MPS bond-dimension sweep, N=6
fixed, chi in {1,2,4,8}, two seeds each. Same style as the finite-size figure
(no star markers, dashed lines for each run's critical epoch, solid line for
the per-chi mean)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "bond_dimension_phase_transition" / "bond_dimension_scaling.json"
OUTPUT_PATHS = [
    ROOT / "latex_outputs" / "paper_latex" / "figures" / "bond_dimension_phase_transition.pdf",
    ROOT / "Main2" / "newHVK" / "results" / "bond_dimension_phase_transition" / "bond_dimension_phase_transition.png",
]

BOND_DIMS = [1, 2, 4, 8]
COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:purple"}
SHADES = {
    1: ["#1f77b4", "#7fb3d8"],
    2: ["#ff7f0e", "#ffbb78"],
    4: ["#2ca02c", "#98df8a"],
    8: ["#9467bd", "#c5b0d5"],
}


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, ax = plt.subplots(figsize=(8, 5.2))

    run_index = {chi: 0 for chi in BOND_DIMS}
    for row in sorted(rows, key=lambda r: (r["bond_dim"], r["dataset"], r["seed"])):
        chi = row["bond_dim"]
        trace = row["order_trace"]
        epochs = list(range(len(trace)))
        color = SHADES[chi][run_index[chi] % len(SHADES[chi])]
        run_index[chi] += 1
        label = f"chi={chi} (seed={row['seed']})"
        ax.plot(epochs, trace, color=color, alpha=0.75, linewidth=1.2, label=label)
        if row["detected"]:
            tc = row["critical_epoch"]
            ax.axvline(tc, color=color, linestyle="--", alpha=0.5, linewidth=0.9)

    for chi in BOND_DIMS:
        detected = [r["critical_epoch"] for r in rows if r["bond_dim"] == chi and r["detected"]]
        if detected:
            mean_tc = sum(detected) / len(detected)
            ax.axvline(mean_tc, color=COLORS[chi], linestyle="-", linewidth=2.2, alpha=0.9,
                       label=f"chi={chi} mean $t_c$={mean_tc:.0f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Order parameter $M_z$")
    ax.set_title("MPS bond-dimension comparison: order parameter vs epoch (N=6 fixed)\ndashed lines mark each run's detected critical epoch")
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
