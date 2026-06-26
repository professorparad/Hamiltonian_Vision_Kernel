# MLP Baseline

This folder contains a fully connected autoencoder for 32x32 grayscale CIFAR
reconstruction.

The image is treated as a 1024-value vector:

```text
1024 pixels -> hidden layers -> latent vector -> 1024 pixels
```

This baseline is intentionally simple. It has no spatial inductive bias, so it
helps separate "can reconstruct at all" from "uses local image structure well".

## Run

```powershell
python Baselines\cifar10_comparisons\mlp\run_mlp_cifar32.py --count 5 --epochs 200 --device cuda
```

Through the combined runner:

```powershell
python Baselines\cifar10_comparisons\main.py --methods mlp --count 5 --epochs 200 --device cuda
```

## Outputs

The script writes:

- `outputs/mlp_cifar32_metrics.csv`
