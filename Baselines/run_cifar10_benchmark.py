from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_DATASET_DIR = REPO_ROOT / "Baselines" / "datasets" / "cifar10_subset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Baselines" / "outputs" / "cifar10_benchmark"


def psnr_from_mse(value: float) -> float:
    if value <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(value))


def run_command(command: list[str], done_file: Path) -> None:
    if done_file.exists():
        print(f"Skipping completed run: {done_file.parent}")
        return
    done_file.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    done_file.write_text("ok\n", encoding="utf-8")


def read_hvk_metrics(csv_path: Path, model: str, image_name: str) -> dict:
    rows = pd.read_csv(csv_path)
    final = rows.iloc[-1]
    best_index = rows["reconstruction_mse"].idxmin()
    best = rows.loc[best_index]
    phase_epoch = rows["order_parameter_susceptibility"].idxmax()
    critical_epoch = int(rows.loc[phase_epoch, "epoch"])
    final_mse = float(final["reconstruction_mse"])
    return {
        "model": model,
        "image": image_name,
        "final_mse": final_mse,
        "best_mse": float(best["reconstruction_mse"]),
        "best_epoch": int(best["epoch"]),
        "psnr": psnr_from_mse(final_mse),
        "final_energy": float(final["mean_energy"]),
        "critical_epoch": critical_epoch,
        "ssim": "",
    }


def read_gan_metrics(summary_path: Path, history_path: Path, image_name: str) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path)
    best_index = history["reconstruction_mse"].idxmin()
    best = history.loc[best_index]
    return {
        "model": "GAN",
        "image": image_name,
        "final_mse": float(summary["metrics"]["mse"]),
        "best_mse": float(best["reconstruction_mse"]),
        "best_epoch": int(best["epoch"]),
        "psnr": float(summary["metrics"]["psnr"]),
        "final_energy": "",
        "critical_epoch": "",
        "ssim": float(summary["metrics"]["ssim"]),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    aggregate_rows = []
    for model in ["1D HVK", "2D HVK", "GAN"]:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        aggregate_row = {"model": model, "images": len(model_rows)}
        for metric in ["final_mse", "best_mse", "psnr"]:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=float)
            aggregate_row[f"mean_{metric}"] = float(values.mean())
            aggregate_row[f"std_{metric}"] = float(values.std(ddof=0))
        numeric_energy = [
            float(row["final_energy"]) for row in model_rows if row["final_energy"] != ""
        ]
        aggregate_row["mean_final_energy"] = (
            float(np.mean(numeric_energy)) if numeric_energy else ""
        )
        numeric_critical = [
            float(row["critical_epoch"])
            for row in model_rows
            if row["critical_epoch"] != ""
        ]
        aggregate_row["mean_critical_epoch"] = (
            float(np.mean(numeric_critical)) if numeric_critical else ""
        )
        numeric_ssim = [float(row["ssim"]) for row in model_rows if row["ssim"] != ""]
        aggregate_row["mean_ssim"] = float(np.mean(numeric_ssim)) if numeric_ssim else ""
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> None:
    image_paths = sorted((args.dataset_dir / "images").glob("*.png"))[: args.count]
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under {args.dataset_dir / 'images'}")

    for image_path in image_paths:
        stem = image_path.stem
        run_command(
            [
                str(PYTHON),
                "Main/main.py",
                "--image-path",
                str(image_path),
                "--output-dir",
                str(args.output_dir / "hvk1d" / stem),
                "--steps",
                str(args.epochs),
                "--device",
                args.device,
                "--no-epoch-media",
            ],
            args.output_dir / "hvk1d" / stem / ".done",
        )
        run_command(
            [
                str(PYTHON),
                "Main2/main.py",
                "--image-path",
                str(image_path),
                "--output-dir",
                str(args.output_dir / "hvk2d" / stem),
                "--steps",
                str(args.epochs),
                "--device",
                args.device,
                "--no-gif",
            ],
            args.output_dir / "hvk2d" / stem / ".done",
        )
        run_command(
            [
                str(PYTHON),
                "Baselines/gan_reconstruction/run_gan.py",
                "--image-path",
                str(image_path),
                "--output-dir",
                str(args.output_dir / "gan" / stem),
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
            ],
            args.output_dir / "gan" / stem / ".done",
        )

    rows = []
    for image_path in image_paths:
        stem = image_path.stem
        rows.append(
            read_hvk_metrics(
                args.output_dir / "hvk1d" / stem / "hvk_epoch_reconstruction_table.csv",
                "1D HVK",
                image_path.name,
            )
        )
        rows.append(
            read_hvk_metrics(
                args.output_dir / "hvk2d" / stem / "hvk_epoch_reconstruction_table.csv",
                "2D HVK",
                image_path.name,
            )
        )
        rows.append(
            read_gan_metrics(
                args.output_dir / "gan" / stem / "gan_summary.json",
                args.output_dir / "gan" / stem / "gan_training_history.csv",
                image_path.name,
            )
        )

    aggregate_rows = aggregate(rows)
    write_csv(args.output_dir / "cifar10_per_image_metrics.csv", rows)
    write_csv(args.output_dir / "cifar10_aggregate_metrics.csv", aggregate_rows)
    (args.output_dir / "cifar10_aggregate_metrics.json").write_text(
        json.dumps(aggregate_rows, indent=2), encoding="utf-8"
    )
    print(f"Wrote per-image metrics to {args.output_dir / 'cifar10_per_image_metrics.csv'}")
    print(f"Wrote aggregate metrics to {args.output_dir / 'cifar10_aggregate_metrics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 200-epoch 1D HVK, 2D HVK, and GAN benchmarks on CIFAR-10 subset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
