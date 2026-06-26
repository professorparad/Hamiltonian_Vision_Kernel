# Main HVK1D Pipeline

`Main/` contains the primary HVK1D reconstruction pipeline. This is the version
used for the standard chain model and the U(1)-symmetric chain model.

The entry point is:

```powershell
python Main\main.py --device auto
```

## What Happens In A Run

The training script:

1. Loads a grayscale image, usually `Main/data/monalisa.jpg`.
2. Splits the image into square patches.
3. Extracts MPS-style features from each patch.
4. Builds sinusoidal positional encodings for patch locations.
5. Sends features and positions into a PennyLane variational circuit.
6. Computes observables and a Hamiltonian energy term.
7. Trains a patch decoder with reconstruction loss plus energy regularization.
8. Stitches reconstructed patches back into an image.
9. Optionally saves plots, GIFs, order-parameter curves, and metrics.

## Variants

`Main/main.py` supports:

- `standard`: learnable `Jx`, `Jy`, and `Jz` couplings.
- `symmetric`: U(1)-symmetric coupling structure, `J * ZZ + K * (XX + YY)`.
- `both`: runs standard and symmetric side by side and writes comparison output.

Example:

```powershell
python Main\main.py --model-variant both --steps 120 --device auto
```

## Source Layout

- `main.py`: CLI wrapper and variant comparison logic.
- `src/training/`: dataset construction, training loop, phase media, and order
  parameter tracking.
- `src/quantum/`: PennyLane circuits and HVK quantum model classes.
- `src/tensornetworks/`: MPS feature extraction and reconstruction helpers.
- `src/preprocessing/`: image loading, patch extraction, and positional encoding.
- `src/decoder/`: neural patch decoder.
- `src/reconstruction/`: patch stitching and seam blending.
- `src/visualization/`: plotting helpers.
- `src/config/training_config.json`: default run configuration.

## Device Behavior

Use `--device auto` unless you are debugging.

The Torch parts can use CUDA when available. The PennyLane quantum simulator
uses CUDA only when `lightning.gpu` is installed. Otherwise the code prefers
`lightning.qubit` on CPU, then falls back to `default.qubit`.

This avoids the common failure mode where Torch tensors are on CUDA but the
PennyLane simulator is still CPU-only.

## Outputs

Default outputs go to:

```text
Main/outputs/training_analysis/
```

Typical files include reconstructions, training curves, order-parameter plots,
phase-transition GIFs, and JSON summaries.

Use `--no-save` for quick debugging:

```powershell
python Main\main.py --steps 1 --no-save --model-variant standard --device auto
```
