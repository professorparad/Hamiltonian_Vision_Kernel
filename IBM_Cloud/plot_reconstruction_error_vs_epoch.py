from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "Main2"
    / "outputs"
    / "training_analysis"
    / "hvk_epoch_reconstruction_table.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "Main2"
    / "outputs"
    / "training_analysis"
    / "reconstruction_error_vs_epoch.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reconstruction MSE vs epoch from an HVK training table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Training table not found: {args.input}")

    rows = pd.read_csv(args.input)
    required = {"epoch", "reconstruction_mse"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(rows["epoch"], rows["reconstruction_mse"], color="tab:blue", linewidth=2)
    ax.set_title("Reconstruction Error vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved reconstruction error plot: {args.output}")


if __name__ == "__main__":
    main()
