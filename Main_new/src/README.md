# Main Source Package

This folder holds the implementation used by `Main/main.py`. The code is split
by role rather than by experiment.

## Subpackages

- `config/`: JSON defaults for the main training script.
- `decoder/`: neural decoder that maps circuit observables back to image
  patches.
- `preprocessing/`: image loading, patch extraction, and positional encoding.
- `quantum/`: PennyLane circuit definitions plus the standard and symmetric
  quantum model modules.
- `reconstruction/`: patch stitching and seam blending.
- `tensornetworks/`: MPS-style feature extraction and reconstruction helpers.
- `training/`: dataset preparation, optimizer loop, order parameters, phase
  media, and output generation.
- `visualization/`: plotting helpers used by the training outputs.

## Notes For Editing

Most model behavior is controlled by `training/training.py`, `quantum/circuit.py`,
`quantum/quantum_model.py`, `quantum/symmetric_model.py`, and
`decoder/patch_decoder.py`.

If you change the public behavior here, check the mirrored package under
`python_library/src/hvk/` as well. That copy is what users would import if the
package is installed.
