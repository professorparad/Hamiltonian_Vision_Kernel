# HVK1D CIFAR Runner

This folder runs the standard 1D Hamiltonian Vision Kernel on 32x32 grayscale
CIFAR images.

The runner uses the core HVK1D model from `Main/src/quantum/quantum_model.py`,
but with CIFAR-specific settings:

- image size: `32`
- patch size: `8`
- patch stride: `4`
- MPS sites: `6`
- positional dimension: `4`
- default epochs: `200`

## Run Directly

```powershell
python Baselines\cifar10_comparisons\hvk1d\run_hvk1d_cifar32.py --count 5 --epochs 200 --device auto
```

Most of the time, run it through the benchmark aggregator instead:

```powershell
python Baselines\cifar10_comparisons\main.py --methods hvk1d --count 5 --epochs 200 --device auto
```

## Device Notes

This method needs PennyLane for the quantum circuit. If `lightning.gpu` is not
installed, `--device auto` will use CPU for the HVK path. Forcing
`--device cuda` without `lightning.gpu` should fail early.

## Outputs

The script writes:

- `outputs/hvk1d_cifar32_metrics.csv`
- per-image reconstruction comparison PNGs
- per-image order-parameter CSVs
- per-image order-parameter plots

The benchmark aggregator copies useful files into the shared
`Baselines/cifar10_comparisons/outputs/` folder.
