# Mona Lisa Comparisons

This folder reuses the CIFAR comparison runner on a single 32x32 grayscale
version of `Main/data/monalisa.jpg`. It is useful when you want a recognizable
image instead of CIFAR samples, but still want the same metric and output layout.

The script prepares a one-image dataset under `datasets/images/monalisa.png`,
then calls `Baselines/cifar10_comparisons/main.py` with `--skip-download`.

## Run All Methods

```powershell
python Baselines\monalisa_comparisons\main.py --methods all --epochs 200 --device auto
```

## Run Selected Methods

```powershell
python Baselines\monalisa_comparisons\main.py --methods hvk1d hvk2d symmetric --epochs 200 --device auto
```

```powershell
python Baselines\monalisa_comparisons\main.py --methods cnn mlp autoencoder gan phl --epochs 200 --device cuda
```

## Useful Arguments

- `--image-path`: source image. Defaults to `Main/data/monalisa.jpg`.
- `--image-size`: resize target. Defaults to `32`.
- `--epochs`: training epochs per method.
- `--device`: `auto`, `cpu`, or `cuda`.
- `--methods`: same method names used by the CIFAR benchmark.

## Outputs

Outputs are written under `Baselines/monalisa_comparisons/outputs/`:

- `monalisa_per_image_metrics.csv`
- `monalisa_aggregate_metrics.csv`
- `monalisa_aggregate_metrics.json`
- `monalisa_metric_comparison.png`
- `visuals/<method>/`
- `per_method_metrics/<method>/`

This suite is mainly a sanity and visualization benchmark. CIFAR is better for
comparing averages across multiple images.
