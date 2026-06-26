# Patch Decoder

This folder contains the neural decoder used by HVK1D.

The decoder takes quantum observables plus positional information and predicts
image patches. It is intentionally small so the quantum feature path remains
visible in the experiment.

Main file:

- `patch_decoder.py`

The decoder is trained jointly with the quantum model parameters.
