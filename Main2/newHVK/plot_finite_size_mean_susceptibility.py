"""Single-plot, three-curve view of the order-parameter change magnitude
(susceptibility) across the qubit-count sweep: for each N, average the
per-run |dM_z/dt| trace across all runs at that N (ensemble mean, computed
before peak-finding, unlike the per-run-then-averaged critical epochs in
Table VIII), then plot the three resulting mean-susceptibility curves
together. This is a complementary statistical view, not a replacement for
the per-run detection table: peak position in the ensemble mean need not
match the mean of individually-detected per-run peaks when per-run traces
are noisy."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "Main2" / "newHVK" / "results" / "finite_size_phase_transition" / "finite_size_scaling.json"
OUTPUT_PATHS = [
    ROOT / "latex_outputs" / "paper_latex" / "figures" / "finite_size_mean_susceptibility.pdf",
    ROOT / "Main2" / "newHVK" / "results" / "finite_size_phase_transition" / "finite_size_mean_susceptibility.png",
]

COLORS = {4: "tab:blue", 6: "tab:red", 8: "tab:green"}


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())
    fig, ax = plt.subplots(figsize=(8, 5.2))

    summary = {}
    for n in (4, 6, 8):
        traces = np.array([r["order_trace"] for r in rows if r["qubit_count"] == n])
        diffs = np.abs(np.diff(traces, axis=1, prepend=traces[:, :1]))
        mean_diff = diffs.mean(axis=0)
        std_diff = diffs.std(axis=0)
        epochs = np.arange(len(mean_diff))
        peak_epoch = int(mean_diff.argmax())
        summary[n] = peak_epoch

        ax.plot(epochs, mean_diff, color=COLORS[n], linewidth=2, label=f"N={n} (peak at t={peak_epoch})")
        ax.fill_between(epochs, mean_diff - std_diff, mean_diff + std_diff, color=COLORS[n], alpha=0.15)
        ax.axvline(peak_epoch, color=COLORS[n], linestyle="--", alpha=0.6, linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Mean susceptibility $\langle|\Delta M_z(t)|\rangle$ (across runs, $\pm$std shaded)")
    ax.set_title("Order-parameter change magnitude vs epoch, qubit-count sweep\nensemble mean across 4 runs per N (2 datasets x 2 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    plt.close(fig)
    print("Peak epochs:", summary)


if __name__ == "__main__":
    main()
