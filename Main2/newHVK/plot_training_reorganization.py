"""Regenerate the manuscript's finite-size training-reorganization figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "results/full_ablation_suite/order_parameter_curve.csv"
OUTPUT = ROOT / "latex_outputs/paper_latex/figures/phase_transition_single_display.pdf"
FIGURES = OUTPUT.parent


def plot_trace(order: np.ndarray, output_name: str, title: str) -> None:
    susceptibility = np.r_[0.0, np.abs(np.diff(order))]
    epochs = np.arange(order.size)
    peak_index = int(np.argmax(susceptibility))
    fig, left = plt.subplots(figsize=(8.2, 3.25))
    right = left.twinx()
    left.plot(epochs, order, color="#285f9e", linewidth=2)
    right.plot(
        epochs, susceptibility,
        color="#b44b3f", linewidth=1.8, linestyle="--",
    )
    left.axvline(peak_index, color="#777777", linewidth=1.1, linestyle=":")
    right.annotate(
        f"Peak susceptibility\n(candidate epoch {peak_index})",
        xy=(peak_index, susceptibility[peak_index]),
        xytext=(0.60, 0.82), textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=9, ha="left", va="center",
    )
    left.set_xlabel("Training epoch")
    left.set_ylabel(r"Order parameter $M_z(t)$", color="#285f9e")
    right.set_ylabel(r"Susceptibility $\mathcal{X}(t)$", color="#b44b3f")
    left.set_title(title)
    left.grid(alpha=0.22)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_multi_dataset_gallery() -> None:
    source = Path(__file__).resolve().parent / "results/phase_transition_multi_dataset"
    labels = {
        "cifar10": ["CIFAR-10: cat", "CIFAR-10: ship (hydrofoil)", "CIFAR-10: ship (sea boat)"],
        "mnist": ["MNIST: digit 5", "MNIST: digit 0"],
        "fashion-mnist": ["Fashion-MNIST: ankle boot", "Fashion-MNIST: T-shirt/top"],
        "pathmnist": ["PathMNIST: adipose", "PathMNIST: mucus"],
        "bloodmnist": ["BloodMNIST: platelet", "BloodMNIST: immature granulocyte"],
        "pneumoniamnist": ["PneumoniaMNIST: positive 1", "PneumoniaMNIST: positive 2"],
    }
    traces: list[tuple[str, np.ndarray]] = []
    for dataset in labels:
        filename = source / f"{dataset}_order_traces.json"
        with filename.open(encoding="utf-8") as handle:
            dataset_traces = json.load(handle)
        traces.extend((label, np.asarray(trace, dtype=float)) for label, trace in zip(labels[dataset], dataset_traces))

    fig, axes = plt.subplots(4, 4, figsize=(11.2, 8.2))
    for axis, (label, order) in zip(axes.flat, traces):
        susceptibility = np.r_[0.0, np.abs(np.diff(order))]
        peak = int(np.argmax(susceptibility))
        twin = axis.twinx()
        axis.plot(np.arange(order.size), order, color="#285f9e", linewidth=1.2)
        twin.plot(np.arange(order.size), susceptibility, color="#b44b3f", linewidth=1.0, linestyle="--")
        axis.axvline(peak, color="#777777", linewidth=0.8, linestyle=":")
        axis.set_title(f"{label}\npeak epoch {peak}", fontsize=8)
        axis.tick_params(labelsize=7)
        twin.tick_params(labelsize=7, colors="#b44b3f")
        axis.grid(alpha=0.16)
    for axis in axes.flat[len(traces):]:
        axis.axis("off")
    fig.suptitle("Per-image order-parameter reorganization diagnostics (13 sampled images)", fontsize=13)
    fig.supxlabel("Training epoch")
    fig.supylabel(r"Order parameter $M_z(t)$")
    fig.text(0.985, 0.5, r"Susceptibility $\mathcal{X}(t)$", rotation=90, va="center", ha="right", color="#b44b3f")
    fig.tight_layout(rect=(0.025, 0.025, 0.975, 0.965))
    fig.savefig(FIGURES / "multi_dataset_phase_transition_gallery.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(DATA)
    curve = data[data["model"] == "HVK2D-entangling-observables"].copy()
    peak = curve.loc[curve["susceptibility"].idxmax()]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), sharex=True)

    axes[0].plot(curve["epoch"], curve["order_parameter"], color="#285f9e", linewidth=2.2)
    axes[0].set_ylabel(r"Order parameter $m(t)$")
    axes[0].set_title("Learned observable order")

    axes[1].plot(curve["epoch"], curve["susceptibility"], color="#b44b3f", linewidth=2.2, linestyle="--")
    axes[1].axvline(peak["epoch"], color="#777777", linewidth=1.2, linestyle=":")
    axes[1].scatter([peak["epoch"]], [peak["susceptibility"]], color="#b44b3f", zorder=3)
    axes[1].annotate(
        f"Peak susceptibility\n(epoch {int(peak['epoch'])})",
        xy=(peak["epoch"], peak["susceptibility"]),
        xytext=(52, 0.165),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        ha="left",
        va="center",
        fontsize=9,
    )
    axes[1].set_ylabel(r"Susceptibility $\mathcal{X}(t)$")
    axes[1].set_title("Finite-size training diagnostic")

    for axis in axes:
        axis.set_xlabel("Training epoch")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("HVK2D order-parameter reorganization during training", fontsize=12)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)

    results = Path(__file__).resolve().parent / "results"
    monalisa_rows = pd.read_csv(
        results / "ablation_study/legacy_hvk_controls/eval_controls/shared-baseline-seed-42/hvk_epoch_correlation_table.csv"
    )
    monalisa_order = (
        monalisa_rows.groupby("epoch", sort=True)["order_parameter"].mean().to_numpy(dtype=float)
    )
    plot_trace(
        monalisa_order,
        "phase_transition_monalisa_single.pdf",
        "Order-parameter reorganization: Monalisa reconstruction (HVK1D)",
    )
    with (results / "phase_transition_multi_dataset/cifar10_order_traces.json").open(encoding="utf-8") as handle:
        cifar_traces = json.load(handle)
    plot_trace(
        np.asarray(cifar_traces[0], dtype=float),
        "phase_transition_cifar_cat_single.pdf",
        'Order-parameter reorganization: CIFAR-10 "cat" reconstruction (HVK2D)',
    )
    plot_multi_dataset_gallery()


if __name__ == "__main__":
    main()
