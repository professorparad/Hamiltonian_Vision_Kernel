"""
Single entry point for Mona Lisa baseline comparisons.

This reuses the CIFAR-32 comparison engine on a one-image 32x32 Mona Lisa
dataset so the metrics, plots, and folder layout match the CIFAR run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_ROOT = Path(__file__).resolve().parent
CIFAR_MAIN = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "main.py"
DEFAULT_IMAGE_PATH = REPO_ROOT / "Main" / "data" / "monalisa.jpg"
DEFAULT_DATASET_DIR = THIS_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = THIS_ROOT / "outputs"


def prepare_monalisa_dataset(image_path: Path, dataset_dir: Path, image_size: int) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = images_dir / "monalisa.png"
    cv2.imwrite(str(output_path), resized)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mona Lisa HVK, classical, GAN, and PHL comparisons."
    )
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to run, same names as Baselines/cifar10_comparisons/main.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_monalisa_dataset(args.image_path, args.dataset_dir, args.image_size)
    print(f"Mona Lisa dataset image: {prepared}", flush=True)

    subprocess.run(
        [
            sys.executable,
            str(CIFAR_MAIN),
            "--dataset-dir",
            str(args.dataset_dir),
            "--output-dir",
            str(args.output_dir),
            "--artifact-prefix",
            "monalisa",
            "--count",
            "1",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--skip-download",
            "--methods",
            *args.methods,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
