"""
Single entry point for CIFAR-10 32x32 baseline comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common import DEFAULT_DATASET_DIR, save_metric_comparison, seed_everything

METHODS = {
    "hvk1d": ("HVK1D", "hvk1d/run_hvk1d_cifar32.py"),
    "hvk2d": ("HVK2D", "hvk2d/run_hvk2d_cifar32.py"),
    "symmetric": ("SymmetricHVK1D", "symmetric_hvk1d/run_symmetric_hvk1d_cifar32.py"),
    "gan": ("GAN", "gan/run_gan_cifar32.py"),
    "phl": ("PHL", "phl/run_phl_cifar32.py"),
    "mlp": ("MLP", "mlp/run_mlp_cifar32.py"),
    "cnn": ("CNN", "cnn/run_cnn_cifar32.py"),
    "autoencoder": ("Autoencoder", "autoencoder/run_autoencoder_cifar32.py"),
}

CSV_GLOBS = {
    "hvk1d": ["hvk1d_cifar32_metrics.csv"],
    "hvk2d": ["hvk2d_cifar32_metrics.csv"],
    "symmetric": ["symmetric_cifar32_metrics.csv", "*cifar32*.csv"],
    "gan": ["gan_cifar32_metrics.csv"],
    "phl": ["phl_cifar32_metrics.csv"],
    "mlp": ["mlp_cifar32_metrics.csv"],
    "cnn": ["cnn_cifar32_metrics.csv"],
    "autoencoder": ["autoencoder_cifar32_metrics.csv"],
}


def output_folder_for(method_key: str) -> Path:
    folder = "symmetric_hvk1d" if method_key == "symmetric" else method_key
    return BENCH_ROOT / folder / "outputs"


def method_output_key(method_key: str) -> str:
    return "symmetric_hvk1d" if method_key == "symmetric" else method_key


def collect_method_outputs(method_key: str, output_root: Path) -> None:
    source_dir = output_folder_for(method_key)
    if not source_dir.exists():
        return

    method_key = method_output_key(method_key)
    visuals_dir = output_root / "visuals" / method_key
    metrics_dir = output_root / "per_method_metrics" / method_key
    for directory in (visuals_dir, metrics_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        target_dir = metrics_dir if path.suffix.lower() in {".csv", ".json"} else visuals_dir
        target_path = target_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_path)


def save_method_metric_summary(rows: list[dict], method_key: str, output_root: Path) -> None:
    if not rows:
        return
    metric_keys = ["mse", "psnr", "ssim", "dice", "iou", "pixel_accuracy"]
    means = {}
    for key in metric_keys:
        values = numeric_values(rows, key)
        if values.size:
            means[key.upper()] = float(values.mean())
    if not means:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_root / "visuals" / method_output_key(method_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(means.keys())
    values = [means[label] for label in labels]

    fig, ax = plt.subplots(figsize=(max(5.0, len(labels) * 1.25), 3.8))
    ax.bar(labels, values, color="tab:blue")
    ax.set_title(f"{METHODS[method_key][0]} mean metrics")
    ax.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_summary.png", dpi=160)
    plt.close(fig)


def run_method(
    method_key: str,
    dataset_dir: Path,
    count: int,
    epochs: int,
    device: str,
    output_root: Path,
) -> list[dict]:
    name, script_rel = METHODS[method_key]
    script = BENCH_ROOT / script_rel
    scratch_output = output_folder_for(method_key)
    if scratch_output.exists():
        shutil.rmtree(scratch_output)
    print(f"\n{'=' * 72}", flush=True)
    print(f"Running {name}", flush=True)
    print(f"{'=' * 72}", flush=True)

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-dir",
            str(dataset_dir),
            "--count",
            str(count),
            "--epochs",
            str(epochs),
            "--device",
            device,
        ],
        check=True,
    )

    output_dir = output_folder_for(method_key)
    for pattern in CSV_GLOBS[method_key]:
        csv_files = sorted(output_dir.glob(pattern))
        if csv_files:
            rows = read_csv_rows(csv_files[0])
            collect_method_outputs(method_key, output_root)
            save_method_metric_summary(rows, method_key, output_root)
            return rows
    print(f"WARNING: no output CSV found for {name} in {output_dir}")
    collect_method_outputs(method_key, output_root)
    return []


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["model", "image", "image_size", "patch_size", "epochs"]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_values(rows: list[dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return np.asarray(values, dtype=np.float64)


def aggregate_results(all_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in all_rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    metric_keys = [
        "mse",
        "psnr",
        "ssim",
        "dice",
        "iou",
        "pixel_accuracy",
        "boundary_f_score",
    ]
    agg = []
    for model, rows in sorted(grouped.items()):
        summary = {"model": model, "num_images": len(rows)}
        for key in metric_keys:
            values = numeric_values(rows, key)
            if values.size == 0:
                continue
            summary[f"mean_{key}"] = float(values.mean())
            summary[f"std_{key}"] = float(values.std(ddof=0))
            if key == "mse":
                summary["best_mse"] = float(values.min())
            else:
                summary[f"best_{key}"] = float(values.max())
        agg.append(summary)
    return agg


def download_dataset_if_needed(dataset_dir: Path, count: int) -> None:
    dl_script = BENCH_ROOT / "download_cifar32.py"
    print("Checking CIFAR-10 32x32 dataset...")
    subprocess.run(
        [
            sys.executable,
            str(dl_script),
            "--output-dir",
            str(dataset_dir),
            "--count",
            str(max(count * 2, count)),
        ],
        check=True,
    )


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 96}")
    print(
        f"{'Model':<20} {'MSE':<12} {'PSNR':<12} {'SSIM':<12} "
        f"{'Dice':<12} {'IoU':<12}"
    )
    print(f"{'-' * 96}")
    for row in rows:
        print(
            f"{row['model']:<20} "
            f"{row.get('mean_mse', float('nan')):<12.6f} "
            f"{row.get('mean_psnr', float('nan')):<12.2f} "
            f"{row.get('mean_ssim', float('nan')):<12.4f} "
            f"{row.get('mean_dice', float('nan')):<12.4f} "
            f"{row.get('mean_iou', float('nan')):<12.4f}"
        )
    print(f"{'=' * 96}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 32x32 HVK, classical, GAN, and PHL comparisons."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "outputs")
    parser.add_argument(
        "--artifact-prefix",
        default="cifar32",
        help="Filename prefix for combined CSV/JSON/PNG outputs.",
    )
    parser.add_argument("--count", type=int, default=5, help="Images per method.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per method.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()) + ["all"],
        default=["all"],
        help="Methods to run.",
    )
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    if not args.skip_download:
        download_dataset_if_needed(args.dataset_dir, args.count)

    methods_to_run = list(METHODS.keys()) if "all" in args.methods else args.methods
    all_rows = []
    for method_key in methods_to_run:
        all_rows.extend(
            run_method(
                method_key,
                args.dataset_dir,
                args.count,
                args.epochs,
                args.device,
                args.output_dir,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = args.output_dir / f"{args.artifact_prefix}_per_image_metrics.csv"
    write_csv(per_image_path, all_rows)

    agg = aggregate_results(all_rows)
    agg_path = args.output_dir / f"{args.artifact_prefix}_aggregate_metrics.csv"
    write_csv(agg_path, agg)
    (args.output_dir / f"{args.artifact_prefix}_aggregate_metrics.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )
    comparison_plot = args.output_dir / f"{args.artifact_prefix}_metric_comparison.png"
    save_metric_comparison(agg, comparison_plot)
    summary_visuals = args.output_dir / "visuals" / "_summary"
    summary_visuals.mkdir(parents=True, exist_ok=True)
    if comparison_plot.exists():
        shutil.copy2(comparison_plot, summary_visuals / "metric_comparison.png")

    print(f"\nPer-image metrics: {per_image_path}")
    print(f"Aggregate metrics: {agg_path}")
    print(f"Aggregate metric plot: {comparison_plot}")
    print(f"Organized visuals: {args.output_dir / 'visuals'}")
    print(f"Per-method CSV/JSON: {args.output_dir / 'per_method_metrics'}")
    print_summary(agg)


if __name__ == "__main__":
    main()
