# Main2 Source Package

This folder contains the implementation of the 2D-grid HVK model.

## Files

- `config.py`: dataclass defaults for the 2D run.
- `dataset.py`: image loading, patching, MPS features, and device transfer.
- `model.py`: PennyLane grid circuit, quantum model, and patch decoder.
- `training.py`: optimizer loop and run orchestration.
- `analysis.py`: order-parameter and phase-transition analysis.
- `outputs.py`: plotting, image saving, and metric output helpers.
- `pathing.py`: path helpers.

The 2D model is kept separate from `Main/src` because the circuit topology and
observables are different enough to make shared code harder to read.
