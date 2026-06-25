# CIFAR-10 32x32 Benchmarks

This folder contains native 32x32 grayscale CIFAR reconstruction benchmarks.

## Layout

- `common.py` shared loading, metrics, plotting, and order-parameter helpers.
- `download_cifar32.py` downloads a small CIFAR-10 subset into `datasets/`.
- `main.py` runs every model and writes aggregate metrics/visualizations.
- `hvk1d/`, `hvk2d/`, `symmetric_hvk1d/` quantum HVK runners.
- `gan/`, `phl/`, `mlp/`, `cnn/`, `autoencoder/` baseline runners.

Generated files are written to `outputs/` folders and are ignored by git.

## Quick Smoke Test

Run this first when changing benchmark code:

```bash
.venv/bin/python Baselines/cifar10_comparisons/smoke_test.py --epochs 1
```

It creates a temporary 32x32 image, runs the comprehensive benchmark, and checks that
all scripts can execute.

## Real CIFAR Run

Download a small CIFAR-10 subset:

```bash
.venv/bin/python Baselines/cifar10_comparisons/download_cifar32.py --count 10
```

Run all models:

```bash
.venv/bin/python Baselines/cifar10_comparisons/main.py --count 5 --epochs 200 --device cpu
```

Run only selected models:

```bash
.venv/bin/python Baselines/cifar10_comparisons/main.py --methods hvk1d hvk2d symmetric phl gan --count 3 --epochs 200 --device cpu --skip-download
```

Quantum runners write per-image order-parameter CSV/PNG files. PHL writes
per-image segmentation visuals in `phl/outputs/`. The combined outputs are:

- `outputs/cifar32_per_image_metrics.csv`
- `outputs/cifar32_aggregate_metrics.csv`
- `outputs/cifar32_aggregate_metrics.json`
- `outputs/cifar32_metric_comparison.png`
- `outputs/visuals/<method>/`
- `outputs/per_method_metrics/<method>/`
