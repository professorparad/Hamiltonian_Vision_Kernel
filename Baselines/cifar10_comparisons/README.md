# CIFAR-10 32x32 Benchmarks

This folder contains native 32x32 grayscale CIFAR reconstruction benchmarks.

## Layout

- `common.py` shared loading, metrics, plotting, and order-parameter helpers.
- `download_cifar32.py` downloads a small CIFAR-10 subset into `datasets/`.
- `run_comprehensive_benchmark.py` runs every model and writes aggregate metrics.
- `hvk1d/`, `hvk2d/`, `symmetric_hvk1d/` quantum HVK runners.
- `gan/`, `mlp/`, `cnn/`, `autoencoder/` classical baseline runners.

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
.venv/bin/python Baselines/cifar10_comparisons/run_comprehensive_benchmark.py --count 5 --epochs 100 --device cpu
```

Run only selected models:

```bash
.venv/bin/python Baselines/cifar10_comparisons/run_comprehensive_benchmark.py --methods hvk1d hvk2d symmetric --count 3 --epochs 50 --device cpu --skip-download
```

Quantum runners also write per-image order-parameter CSV/PNG files.
