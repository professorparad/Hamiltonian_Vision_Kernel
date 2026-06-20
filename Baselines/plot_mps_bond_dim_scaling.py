from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = REPO_ROOT / "Main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.image_loader import load_image_grayscale
from src.preprocessing.patching import extract_patches
from src.reconstruction.patch_stitching import stictch_patches
from src.reconstruction.seam_bleading import blend_seams
from src.tensornetworks.mps_reconstruction import mps_reconstruct


DEFAULT_IMAGE_PATH = REPO_ROOT / "Main" / "data" / "monalisa.jpg"
DEFAULT_BOND_DIMS = [1, 2, 4, 8, 16, 32, 64]


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_scaling(
    *,
    image_path: Path,
    image_size: int,
    patch_size: int,
    bond_dims: list[int],
) -> list[dict]:
    image = load_image_grayscale(str(image_path), size=(image_size, image_size))
    patches, _ = extract_patches(image, patch_size=patch_size)
    rows = []
    for bond_dim in bond_dims:
        reconstructed_patches = np.array(
            [
                [mps_reconstruct(patch, bond_dim=bond_dim, patch_size=patch_size)]
                for patch in patches
            ]
        )
        reconstruction = blend_seams(
            stictch_patches(
                reconstructed_patches,
                image_size=image_size,
                patch_size=patch_size,
            ),
            patch_size=patch_size,
        )
        mse = float(np.mean((reconstruction - image) ** 2))
        rows.append(
            {
                "bond_dim": bond_dim,
                "mse": mse,
                "psnr": psnr_from_mse(mse),
            }
        )
    return rows


def save_plot(path: Path, rows: list[dict], title: str) -> None:
    bond_dims = [row["bond_dim"] for row in rows]
    mse = [row["mse"] for row in rows]
    psnr = [row["psnr"] for row in rows]

    fig, ax_mse = plt.subplots(figsize=(7.2, 4.6))
    ax_mse.plot(bond_dims, mse, marker="o", color="#3b6ea8", label="MSE")
    ax_mse.set_xscale("log", base=2)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("MPS bond dimension")
    ax_mse.set_ylabel("Reconstruction MSE")
    ax_mse.grid(True, which="both", linestyle=":", linewidth=0.7)

    ax_psnr = ax_mse.twinx()
    ax_psnr.plot(bond_dims, psnr, marker="s", color="#d97706", label="PSNR")
    ax_psnr.set_ylabel("PSNR (dB)")

    handles = ax_mse.get_lines() + ax_psnr.get_lines()
    labels = [handle.get_label() for handle in handles]
    ax_mse.legend(handles, labels, loc="best")
    ax_mse.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MPS bond-dimension scaling on one image."
    )
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument(
        "--bond-dims",
        type=int,
        nargs="+",
        default=DEFAULT_BOND_DIMS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_scaling(
        image_path=args.image_path,
        image_size=args.image_size,
        patch_size=args.patch_size,
        bond_dims=args.bond_dims,
    )

    targets = [
        (
            REPO_ROOT
            / "Main"
            / "outputs"
            / "training_analysis"
            / "mps_bond_dim_scaling_1d.csv",
            REPO_ROOT
            / "Main"
            / "outputs"
            / "training_analysis"
            / "mps_bond_dim_scaling_1d.png",
            "1D HVK MPS bond-dimension scaling",
        ),
        (
            REPO_ROOT
            / "Main2"
            / "outputs"
            / "training_analysis"
            / "mps_bond_dim_scaling_2d.csv",
            REPO_ROOT
            / "Main2"
            / "outputs"
            / "training_analysis"
            / "mps_bond_dim_scaling_2d.png",
            "2D HVK MPS bond-dimension scaling",
        ),
    ]
    for csv_path, plot_path, title in targets:
        write_csv(csv_path, rows)
        save_plot(plot_path, rows, title)
        print(f"Wrote {csv_path}")
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
