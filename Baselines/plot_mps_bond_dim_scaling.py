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
) -> tuple[np.ndarray, list[dict], dict[int, np.ndarray]]:
    image = load_image_grayscale(str(image_path), size=(image_size, image_size))
    patches, _ = extract_patches(image, patch_size=patch_size)
    rows = []
    reconstructions = {}
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
        reconstructions[bond_dim] = reconstruction
        mse = float(np.mean((reconstruction - image) ** 2))
        rows.append(
            {
                "bond_dim": bond_dim,
                "mse": mse,
                "psnr": psnr_from_mse(mse),
            }
        )
    return image, rows, reconstructions


def save_plot(
    path: Path,
    original: np.ndarray,
    rows: list[dict],
    reconstructions: dict[int, np.ndarray],
    title: str,
) -> None:
    bond_dims = [row["bond_dim"] for row in rows]
    mse = [row["mse"] for row in rows]
    psnr = [row["psnr"] for row in rows]
    best_row = min(rows, key=lambda row: row["mse"])
    default_row = next((row for row in rows if row["bond_dim"] == 4), rows[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    ax_mse, ax_psnr = axes

    ax_mse.plot(
        bond_dims,
        mse,
        marker="o",
        markersize=8,
        linewidth=2.6,
        color="#16a34a",
        label="Reconstruction MSE",
    )
    ax_mse.fill_between(bond_dims, mse, min(mse), color="#16a34a", alpha=0.14)
    ax_mse.set_xscale("log", base=2)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("MPS bond dimension chi")
    ax_mse.set_xticks(bond_dims)
    ax_mse.set_xticklabels([str(dim) for dim in bond_dims])
    ax_mse.set_ylabel("Reconstruction MSE")
    ax_mse.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax_mse.axvline(4, color="#334155", linestyle="--", linewidth=1.2)
    ax_mse.text(
        4.12,
        default_row["mse"],
        "default chi=4",
        fontsize=9,
        va="bottom",
        color="#334155",
    )

    ax_psnr.plot(
        bond_dims,
        psnr,
        marker="s",
        markersize=8,
        linewidth=2.6,
        color="#2563eb",
        label="PSNR",
    )
    ax_psnr.fill_between(bond_dims, psnr, min(psnr), color="#2563eb", alpha=0.12)
    ax_psnr.set_xscale("log", base=2)
    ax_psnr.set_xlabel("MPS bond dimension chi")
    ax_psnr.set_xticks(bond_dims)
    ax_psnr.set_xticklabels([str(dim) for dim in bond_dims])
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax_psnr.axvline(4, color="#334155", linestyle="--", linewidth=1.2)
    ax_psnr.text(
        4.12,
        default_row["psnr"],
        "default chi=4",
        fontsize=9,
        va="bottom",
        color="#334155",
    )

    for row in rows:
        ax_mse.annotate(
            f"{row['bond_dim']}",
            (row["bond_dim"], row["mse"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1f2937",
        )
        ax_psnr.annotate(
            f"{row['bond_dim']}",
            (row["bond_dim"], row["psnr"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1f2937",
        )

    ax_mse.set_title("MSE vs bond dimension", fontsize=12)
    ax_psnr.set_title("PSNR vs bond dimension", fontsize=12)
    ax_mse.legend(loc="best")
    ax_psnr.legend(loc="best")

    summary = (
        f"Monalisa image | default chi=4: MSE {default_row['mse']:.3e}, "
        f"PSNR {default_row['psnr']:.2f} dB | best chi={best_row['bond_dim']}: "
        f"MSE {best_row['mse']:.3e}, PSNR {best_row['psnr']:.2f} dB"
    )
    fig.suptitle(f"{title}: MPS chi scaling to reconstruction quality", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.02, summary, ha="center", va="top", fontsize=10)
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
    original, rows, reconstructions = evaluate_scaling(
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
        save_plot(plot_path, original, rows, reconstructions, title)
        print(f"Wrote {csv_path}")
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
