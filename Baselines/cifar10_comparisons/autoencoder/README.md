# Patch Autoencoder Baseline

This folder contains a classical patch-based autoencoder. It is meant to be a
closer structural comparison to HVK than the full-image MLP or CNN.

The model:

- splits a 32x32 image into 8x8 patches
- encodes each patch with linear layers
- decodes each patch back to pixels
- stitches patches back into the image grid

No quantum circuit is used here. It is all PyTorch and can run on CUDA when
PyTorch CUDA is installed.

## Run

```powershell
python Baselines\cifar10_comparisons\autoencoder\run_autoencoder_cifar32.py --count 5 --epochs 200 --device cuda
```

Through the combined runner:

```powershell
python Baselines\cifar10_comparisons\main.py --methods autoencoder --count 5 --epochs 200 --device cuda
```

## Outputs

The script writes:

- `outputs/autoencoder_cifar32_metrics.csv`
