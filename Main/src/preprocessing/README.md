# Preprocessing

This folder handles image preparation before HVK training.

## Files

- `image_loader.py`: loads images as grayscale arrays.
- `patching.py`: extracts patches and tracks patch positions.
- `positional_encoding.py`: builds sinusoidal position encodings.

## Notes

Patching and positional encoding are part of the model definition, not just data
loading. If patch size, stride, or positional dimension changes, the feature
space seen by the quantum model changes too.
