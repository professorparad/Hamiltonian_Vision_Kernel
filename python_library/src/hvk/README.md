# HVK Package Source

This is the importable package source for `hvk`.

It mirrors the research scripts in `Main/` and `Main2/`, but with a public API
layer so users can call HVK from Python instead of only through scripts.

## Main Areas

- `api.py`: public entry points and CLI bridge.
- `config.py`: shared run configuration.
- `training/`: HVK1D training implementation.
- `quantum/`: HVK1D PennyLane circuit and models.
- `hvk2d/`: 2D-grid model, config, and training path.
- `decoder/`, `preprocessing/`, `reconstruction/`, `tensornetworks/`,
  `visualization/`: package copies of the core helpers.

When fixing core behavior, keep this package copy in sync with the source
scripts unless there is a clear reason not to.
