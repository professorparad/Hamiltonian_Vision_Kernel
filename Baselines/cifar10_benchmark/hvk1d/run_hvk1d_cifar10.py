from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    PYTHON,
    image_paths,
    read_hvk_metrics,
    run_command,
    write_metric_outputs,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run(args: argparse.Namespace) -> None:
    rows = []
    for image_path in image_paths(args.dataset_dir, args.count):
        image_output = args.output_dir / "runs" / image_path.stem
        run_command(
            [
                str(PYTHON),
                "Main/main.py",
                "--image-path",
                str(image_path),
                "--output-dir",
                str(image_output),
                "--steps",
                str(args.epochs),
                "--device",
                args.device,
                "--no-epoch-media",
            ],
            image_output / ".done",
        )
        rows.append(
            read_hvk_metrics(
                image_output / "hvk_epoch_reconstruction_table.csv",
                "1D HVK",
                image_path.name,
            )
        )
    summary = write_metric_outputs(args.output_dir, rows)
    print(f"1D HVK CIFAR-10 summary: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1D HVK on CIFAR-10 subset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
