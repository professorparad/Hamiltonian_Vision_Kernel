# CNN Baseline

This folder contains a compact convolutional autoencoder for 32x32 grayscale
CIFAR reconstruction.

The model is intentionally small:

- three strided convolution blocks for encoding
- one bottleneck projection
- transposed convolutions for decoding
- MSE reconstruction loss

It is a practical sanity baseline: if HVK is doing something useful, it should
be compared against a simple image model like this.

## Run

```powershell
python Baselines\cifar10_comparisons\cnn\run_cnn_cifar32.py --count 5 --epochs 200 --device cuda
```

Through the combined runner:

```powershell
python Baselines\cifar10_comparisons\main.py --methods cnn --count 5 --epochs 200 --device cuda
```

## Outputs

The script writes:

- `outputs/cnn_cifar32_metrics.csv`

The aggregator copies metrics and plots into the shared benchmark output folder.
