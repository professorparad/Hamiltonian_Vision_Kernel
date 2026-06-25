"""
Comprehensive CIFAR-10 benchmark at native 32×32 resolution.
Runs all models and produces a unified comparison CSV.

Models compared:
  1. HVK1D (original 1D chain)
  2. HVK2D (2D grid)
  3. SymmetricHVK1D (U(1) symmetric: J*ZZ + K*(XX+YY))
  4. GAN (patch-based generative adversarial network)
  5. MLP (multilayer perceptron autoencoder)
  6. CNN (convolutional autoencoder)
  7. Autoencoder (patch-based classical autoencoder)
"""
from __future__ import annotations

import argparse
import csv
import json
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
    "mlp": ("MLP", "mlp/run_mlp_cifar32.py"),
    "cnn": ("CNN", "cnn/run_cnn_cifar32.py"),
    "autoencoder": ("Autoencoder", "autoencoder/run_autoencoder_cifar32.py"),
}


def run_method(
    method_key: str,
    dataset_dir: Path,
    count: int,
    epochs: int,
    device: str,
) -> list[dict]:
    """Run a single method and collect results."""
    name, script_rel = METHODS[method_key]
    script = BENCH_ROOT / script_rel
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")

    import subprocess
    import sys as _sys

    cmd = [
        _sys.executable,
        str(script),
        "--dataset-dir", str(dataset_dir),
        "--count", str(count),
        "--epochs", str(epochs),
        "--device", device,
    ]
    subprocess.run(cmd, check=True)

    # Read results from the output CSV
    output_key = "symmetric_hvk1d" if method_key == "symmetric" else method_key
    output_dir = BENCH_ROOT / output_key / "outputs"
    csv_files = sorted(output_dir.glob(f"{method_key}_cifar32_metrics.csv"))
    if not csv_files:
        # Try alternate naming
        csv_files = sorted(output_dir.glob("*cifar32*.csv"))
    if not csv_files:
        print(f"WARNING: No output CSV found for {name}")
        return []

    import pandas as pd
    rows = pd.read_csv(csv_files[0]).to_dict(orient="records")
    return rows


def aggregate_results(all_rows: list[dict]) -> list[dict]:
    """Aggregate per-image results into model-level summaries."""
    df = {}
    for row in all_rows:
        model = row["model"]
        if model not in df:
            df[model] = []
        df[model].append(row)

    agg = []
    for model, rows in sorted(df.items()):
        mse_vals = np.array([r["mse"] for r in rows])
        psnr_vals = np.array([r["psnr"] for r in rows])
        ssim_vals = np.array([r["ssim"] for r in rows])

        agg.append({
            "model": model,
            "num_images": len(rows),
            "mean_mse": float(mse_vals.mean()),
            "std_mse": float(mse_vals.std(ddof=0)),
            "mean_psnr": float(psnr_vals.mean()),
            "std_psnr": float(psnr_vals.std(ddof=0)),
            "mean_ssim": float(ssim_vals.mean()),
            "std_ssim": float(ssim_vals.std(ddof=0)),
            "best_mse": float(mse_vals.min()),
            "best_psnr": float(psnr_vals.max()),
            "best_ssim": float(ssim_vals.max()),
        })
    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive CIFAR-10 benchmark at 32×32."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT)
    parser.add_argument("--count", type=int, default=5, help="Images per model (default=5)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default=100)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()) + ["all"],
        default=["all"],
        help="Which methods to run",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    args = parser.parse_args()

    seed_everything()

    # Download dataset if needed
    if not args.skip_download:
        import subprocess
        dl_script = BENCH_ROOT / "download_cifar32.py"
        if dl_script.exists():
            print("Downloading CIFAR-10 dataset at 32×32...")
            subprocess.run(
                [
                    sys.executable,
                    str(dl_script),
                    "--output-dir",
                    str(args.dataset_dir),
                    "--count",
                    str(args.count * 2),
                ],
                check=True,
            )

    # Determine which methods to run
    methods_to_run = (
        list(METHODS.keys()) if "all" in args.methods else args.methods
    )

    all_rows = []
    for method_key in methods_to_run:
        rows = run_method(
            method_key,
            args.dataset_dir,
            args.count,
            args.epochs,
            args.device,
        )
        all_rows.extend(rows)

    # Write per-image metrics
    output_dir = args.output_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "cifar32_per_image_metrics.csv"
    if all_rows:
        with per_image_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-image metrics: {per_image_path}")

    # Aggregate
    agg = aggregate_results(all_rows)
    agg_path = output_dir / "cifar32_aggregate_metrics.csv"
    if agg:
        with agg_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            writer.writeheader()
            writer.writerows(agg)
        print(f"Aggregate metrics: {agg_path}")

        # Also JSON
        agg_json_path = output_dir / "cifar32_aggregate_metrics.json"
        agg_json_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"Aggregate metrics (JSON): {agg_json_path}")
        plot_path = output_dir / "cifar32_metric_comparison.png"
        save_metric_comparison(agg, plot_path)
        print(f"Aggregate metric plot: {plot_path}")

        # Print summary
        print(f"\n{'='*80}")
        print(f"{'Model':<20} {'MSE':<12} {'PSNR':<12} {'SSIM':<12} {'Best MSE':<12}")
        print(f"{'-'*80}")
        for a in agg:
            print(
                f"{a['model']:<20} "
                f"{a['mean_mse']:.6f}    "
                f"{a['mean_psnr']:.2f}     "
                f"{a['mean_ssim']:.4f}    "
                f"{a['best_mse']:.6f}"
            )
        print(f"{'='*80}")
    else:
        print("No results collected. Something went wrong.")


if __name__ == "__main__":
    main()
