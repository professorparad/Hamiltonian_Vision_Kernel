# Tensor-Network Helpers

This folder contains the MPS-style feature and reconstruction helpers used
before the quantum circuit.

## Files

- `mps_features.py`: extracts compact tensor-network features from image
  patches.
- `mps_reconstruction.py`: helper code for reconstructing or inspecting MPS
  representations.

## Why It Matters

The MPS stage gives each patch a structured feature vector before positional
encoding and quantum processing. It is one of the parts that makes HVK different
from a plain neural autoencoder.
