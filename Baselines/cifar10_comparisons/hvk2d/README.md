# HVK2D CIFAR Runner

This folder runs the 2D-grid Hamiltonian Vision Kernel on 32x32 grayscale CIFAR
images. It imports the grid model from `Main2/src/model.py`.

Compared with HVK1D, this model uses a small lattice connectivity pattern:

- horizontal edges
- vertical edges
- lattice ZZ correlations
- one learnable 2D coupling vector

## Run Directly

```powershell
python Baselines\cifar10_comparisons\hvk2d\run_hvk2d_cifar32.py --count 5 --epochs 200 --device auto
```

Through the combined benchmark:

```powershell
python Baselines\cifar10_comparisons\main.py --methods hvk2d --count 5 --epochs 200 --device auto
```

## Device Notes

`--device auto` is recommended. HVK2D uses CUDA only when PennyLane
`lightning.gpu` is installed and working. If not, it uses CPU instead.

The ordinary PyTorch CUDA check is not enough for this method because the quantum
simulation backend has its own dependency stack.

## Outputs

The script writes:

- `outputs/hvk2d_cifar32_metrics.csv`
- per-image reconstruction comparison PNGs
- per-image order-parameter CSVs
- per-image order-parameter plots
