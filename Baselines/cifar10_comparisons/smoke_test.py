"""Run a tiny end-to-end check for the CIFAR-32 benchmark scripts."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent


def create_smoke_dataset(root: Path) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image = np.linspace(0, 255, 32 * 32, dtype=np.uint8).reshape(32, 32)
    cv2.imwrite(str(images_dir / "smoke.png"), image)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the CIFAR-32 benchmark runners.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to run, passed through to main.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="hvk_cifar_smoke_") as tmp:
        dataset_dir = create_smoke_dataset(Path(tmp))
        cmd = [
            sys.executable,
            str(BENCH_ROOT / "main.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--count",
            "1",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--skip-download",
            "--methods",
            *args.methods,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
