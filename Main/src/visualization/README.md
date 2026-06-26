# Visualization Helpers

This folder contains plotting helpers used by the training scripts.

## Files

- `training_curve.py`: loss curves.
- `reconstruction_plots.py`: original/reconstruction comparisons.
- `observable_plots.py`: circuit observable summaries.
- `entropy_maps.py`: entropy-style visual diagnostics.

The training scripts use Matplotlib's non-interactive `Agg` backend so plots can
be generated from terminals, CI jobs, and headless machines.
