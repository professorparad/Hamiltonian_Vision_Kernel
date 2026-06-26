# Baseline Experiments

This folder contains the comparison scripts used to put the HVK variants next to
plain neural and Hamiltonian-inspired baselines. The intent is not to beat every
modern image model. The intent is to keep the comparisons small, repeatable, and
easy to inspect.

## Folders

- `cifar10_comparisons/`: main benchmark suite on grayscale CIFAR-10 images
  resized or kept at 32x32.
- `monalisa_comparisons/`: runs the same comparison layout on the Mona Lisa
  image from `Main/data/monalisa.jpg`.
- `plot_mps_bond_dim_scaling.py`: helper for bond-dimension scaling plots.

## Method Families

The CIFAR suite currently includes:

- `hvk1d`: standard Hamiltonian Vision Kernel chain model.
- `hvk2d`: 2D grid HVK model.
- `symmetric`: U(1)-symmetric HVK1D.
- `cnn`: convolutional autoencoder.
- `mlp`: fully connected autoencoder.
- `autoencoder`: patch-based classical autoencoder.
- `gan`: patch reconstruction GAN.
- `phl`: Parameterized Hamiltonian Learning segmentation baseline.

The HVK methods use PennyLane circuits and may fall back to CPU if
`lightning.gpu` is not installed. The plain Torch methods can use CUDA through
PyTorch.

## Common Runs

Run all CIFAR methods:

```powershell
python Baselines\cifar10_comparisons\main.py --count 5 --epochs 200 --device auto
```

Run only CUDA-friendly Torch baselines:

```powershell
python Baselines\cifar10_comparisons\main.py --methods cnn mlp autoencoder gan phl --count 5 --epochs 200 --device cuda
```

Run only HVK methods:

```powershell
python Baselines\cifar10_comparisons\main.py --methods hvk1d hvk2d symmetric --count 5 --epochs 200 --device auto
```

Run the Mona Lisa comparison:

```powershell
python Baselines\monalisa_comparisons\main.py --methods all --epochs 200 --device auto
```

## Output Layout

Each method writes its own CSV and visual files under its local `outputs/`
folder. The combined runner copies the useful pieces into:

- `outputs/visuals/<method>/`
- `outputs/per_method_metrics/<method>/`
- `outputs/cifar32_per_image_metrics.csv`
- `outputs/cifar32_aggregate_metrics.csv`
- `outputs/cifar32_aggregate_metrics.json`
- `outputs/cifar32_metric_comparison.png`

The Mona Lisa suite uses the same structure, with `monalisa` as the file prefix.
