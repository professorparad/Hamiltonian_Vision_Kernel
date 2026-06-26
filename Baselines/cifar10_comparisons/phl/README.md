# PHL Baseline

This folder contains the Parameterized Hamiltonian Learning baseline. Unlike the
reconstruction autoencoders, PHL is closer to a segmentation-style experiment.

The model learns a small differentiable Ising-style energy:

- unary intensity term
- learnable threshold
- horizontal smoothness coupling
- vertical smoothness coupling

It uses an Otsu mask as a pseudo-target, then reports segmentation metrics as
well as reconstruction-style metrics on the probability map.

## Run

```powershell
python Baselines\cifar10_comparisons\phl\run_phl_cifar32.py --count 5 --epochs 200 --device cuda
```

Through the combined runner:

```powershell
python Baselines\cifar10_comparisons\main.py --methods phl --count 5 --epochs 200 --device cuda
```

## Outputs

The script writes:

- `outputs/phl_cifar32_metrics.csv`
- `outputs/training_history/*_phl_history.csv`
- per-image segmentation visualizations

PHL metrics include Dice, IoU, pixel accuracy, precision, recall, and boundary
F-score.
