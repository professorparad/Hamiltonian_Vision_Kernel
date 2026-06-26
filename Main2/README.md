# Main2 HVK2D Pipeline

`Main2/` contains the 2D-grid version of the Hamiltonian Vision Kernel. It keeps
the same reconstruction goal as `Main/`, but the quantum circuit uses a small
2D lattice pattern instead of a simple 1D chain.

Run it with:

```powershell
python Main2\main.py --device auto
```

## Why This Exists Separately

The 1D and 2D versions share the same broad idea, but the circuit topology and
order-parameter interpretation are different enough that keeping a separate
folder is easier to reason about during experiments.

The 2D model uses:

- horizontal edges
- vertical edges
- lattice ZZ correlations
- a 2D grid Hamiltonian coupling vector

## Source Layout

- `main.py`: CLI entry point.
- `src/config.py`: default run configuration.
- `src/dataset.py`: patch and feature dataset construction.
- `src/model.py`: 2D PennyLane circuit, quantum grid model, and decoder.
- `src/training.py`: training loop and output assembly.
- `src/analysis.py`: order-parameter and phase-transition analysis.
- `src/outputs.py`: plotting and file-writing helpers.
- `Scripts/`: older notebook-era artifacts and saved media from exploratory
  runs.

## Device Behavior

Like the main HVK1D path, `--device auto` is safest. It uses CUDA only when the
relevant backend exists. For the PennyLane quantum simulator, that means
`lightning.gpu`. Without it, the model runs on CPU through `lightning.qubit` or
`default.qubit`.

## Outputs

Generated outputs go to:

```text
Main2/outputs/training_analysis/
```

The `Scripts/` folder contains historical notebook outputs and should not be
treated as the canonical runtime output directory.
