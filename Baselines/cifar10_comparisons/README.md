# CIFAR-10 32x32 Comparisons

This is the main benchmark suite for the repository. It uses small grayscale
CIFAR-10 images at native 32x32 resolution so the whole experiment can run on a
laptop while still giving useful comparisons between HVK and classical models.

The scripts are intentionally plain Python entry points. Each method can be run
by itself, and `main.py` can run a group of methods and collect the results into
one output folder.

## Folder Layout

- `main.py`: orchestrates method runs and aggregates metrics.
- `common.py`: shared device selection, image loading, metrics, plotting, and
  CSV helpers.
- `download_cifar32.py`: creates a small local CIFAR-10 image set.
- `smoke_test.py`: builds a temporary synthetic image and checks that selected
  runners execute.
- `hvk1d/`: standard HVK chain model.
- `hvk2d/`: 2D grid HVK model.
- `symmetric_hvk1d/`: U(1)-symmetric HVK1D.
- `cnn/`, `mlp/`, `autoencoder/`, `gan/`: reconstruction baselines.
- `phl/`: segmentation-style Parameterized Hamiltonian Learning baseline.

## Devices

Use `--device auto` by default. It does the least surprising thing:

- Torch-only methods use CUDA when PyTorch can see a CUDA GPU.
- HVK methods use CUDA only when PennyLane `lightning.gpu` is available.
- If `lightning.gpu` is missing, HVK methods run on CPU instead of mixing CPU
  and CUDA tensors.

If you force `--device cuda` for HVK without `lightning.gpu`, the runner should
fail early with a clear message.

## Quick Checks

Smoke test one classical and one HVK runner:

```powershell
python Baselines\cifar10_comparisons\smoke_test.py --epochs 1 --methods cnn hvk1d --device auto
```

Check CUDA in the active Python environment:

```powershell
python cuda_check.py
```

## Running CIFAR

Download or refresh a small local dataset:

```powershell
python Baselines\cifar10_comparisons\download_cifar32.py --count 10
```

Run all methods:

```powershell
python Baselines\cifar10_comparisons\main.py --count 5 --epochs 200 --device auto
```

Run only the Torch baselines on CUDA:

```powershell
python Baselines\cifar10_comparisons\main.py --methods cnn mlp autoencoder gan phl --count 5 --epochs 200 --device cuda --skip-download
```

Run HVK variants:

```powershell
python Baselines\cifar10_comparisons\main.py --methods hvk1d hvk2d symmetric --count 5 --epochs 200 --device auto --skip-download
```

## Metrics

Reconstruction methods report:

- MSE
- PSNR
- SSIM

PHL also reports segmentation-style metrics:

- Dice
- IoU
- pixel accuracy
- precision
- recall
- boundary F-score

The HVK runners additionally write order-parameter CSV and PNG files per image.

## Outputs

Method-level outputs stay under each method folder. The aggregator writes:

- `outputs/cifar32_per_image_metrics.csv`
- `outputs/cifar32_aggregate_metrics.csv`
- `outputs/cifar32_aggregate_metrics.json`
- `outputs/cifar32_metric_comparison.png`
- `outputs/visuals/<method>/`
- `outputs/per_method_metrics/<method>/`

Generated output folders are analysis artifacts. They should not be treated as
source files.
