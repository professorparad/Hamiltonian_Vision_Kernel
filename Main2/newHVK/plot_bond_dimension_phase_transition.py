"""Order-parameter-vs-epoch figure for the MPS bond-dimension sweep, N=6
fixed, chi in {1,2,4,6,8}. chi=1,2 retain the original 1-image/2-seed scope;
chi=4,6,8 were extended to 5 images x 2 seeds (10 runs each). With that many
runs per chi for 4/6/8, individual-trace plotting (as used for the original
2-seed version) becomes unreadable, so this switches to the ensemble-mean
+/- std-band style already used for the finite-size susceptibility figure,
which scales correctly across the mixed n=2/n=10 group sizes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "bond_dimension_phase_transition" / "bond_dimension_scaling.json"
OUTPUT_PATHS = [
    ROOT / "latex_outputs" / "paper_latex" / "figures" / "bond_dimension_phase_transition.pdf",
    ROOT / "Main2" / "newHVK" / "results" / "bond_dimension_phase_transition" / "bond_dimension_phase_transition.png",
]

BOND_DIMS = [1, 2, 4, 6, 8]
COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 6: "tab:red", 8: "tab:purple"}
LINESTYLES = {1: "-", 2: "--", 4: "-.", 6: ":", 8: (0, (3, 1, 1, 1))}


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, ax = plt.subplots(figsize=(8, 5.2))

    for chi in BOND_DIMS:
        chi_rows = [r for r in rows if r["bond_dim"] == chi]
        if not chi_rows:
            continue
        traces = np.array([r["order_trace"] for r in chi_rows])
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0)
        epochs = np.arange(len(mean_trace))
        n = len(chi_rows)

        detected = [r["critical_epoch"] for r in chi_rows if r["detected"]]
        mean_tc = sum(detected) / len(detected) if detected else None
        label = f"chi={chi} (n={n}, mean $t_c$={mean_tc:.0f})" if mean_tc is not None else f"chi={chi} (n={n})"

        ax.plot(epochs, mean_trace, color=COLORS[chi], linestyle=LINESTYLES[chi], linewidth=2, label=label)
        ax.fill_between(epochs, mean_trace - std_trace, mean_trace + std_trace, color=COLORS[chi], alpha=0.12)
        if mean_tc is not None:
            ax.axvline(mean_tc, color=COLORS[chi], linestyle=":", alpha=0.5, linewidth=1.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Order parameter $M_z(t)$ (mean $\pm$ std across runs)")
    ax.set_title(
        "MPS bond-dimension comparison: order parameter vs epoch (N=6 fixed)\n"
        "chi=1,2: 1 image x 2 seeds; chi=4,6,8: 5 images x 2 seeds"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
