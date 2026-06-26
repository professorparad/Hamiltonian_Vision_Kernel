# Symmetric HVK1D CIFAR Runner

This folder runs the U(1)-symmetric HVK1D variant on 32x32 grayscale CIFAR
images.

The main difference from standard HVK1D is the Hamiltonian coupling structure:

```text
H = sum J_b * ZZ_b + sum K_b * (XX_b + YY_b)
```

That ties the XX and YY couplings together and gives the model an axial symmetry
in the XY plane.

## Run Directly

```powershell
python Baselines\cifar10_comparisons\symmetric_hvk1d\run_symmetric_hvk1d_cifar32.py --count 5 --epochs 200 --device auto
```

Through the combined benchmark:

```powershell
python Baselines\cifar10_comparisons\main.py --methods symmetric --count 5 --epochs 200 --device auto
```

## Device Notes

This is still a PennyLane quantum model. Use `--device auto` unless
`lightning.gpu` is installed. On native Windows, the GPU backend may not install
cleanly because of the cuQuantum dependency stack.

## Outputs

The script writes:

- `outputs/symmetric_cifar32_metrics.csv`
- per-image reconstruction comparison PNGs
- per-image order-parameter CSVs
- per-image order-parameter plots
