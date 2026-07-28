# Training Code

This folder contains the main HVK1D training pipeline.

## Files

- `training.py`: dataset construction, device resolution, model setup, optimizer
  loop, output generation, and CLI support.
- `order_parameters.py`: order-parameter summaries and phase-transition
  detection.
- `phase_media.py`: frame, GIF, and plot helpers for training dynamics.
- `data_generator.py`: structured output writer for epoch-level analysis.

## Notes

The training loop combines reconstruction loss with a small Hamiltonian energy
term. The model and decoder are trained together.

If you change device behavior here, check:

- `Baselines/cifar10_comparisons/common.py`
- `python_library/src/hvk/training/training.py`

Those files should stay conceptually aligned.
