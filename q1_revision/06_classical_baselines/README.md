# Classical baselines

## Reviewer ask

Add: convolutional autoencoder, coordinate MLP, MPS-only model, Fourier-feature MLP,
classical quadratic/polynomial features, tensor-network model with matched latent
dimension and parameter count.

## What already exists (`Baselines/cifar10_comparisons/`)

| Reviewer's ask | Existing folder | Notes |
|---|---|---|
| Convolutional autoencoder | `autoencoder/` | Present, needs checking against current split protocol |
| Coordinate MLP | `mlp/` | Present, but is a *direct image-reconstruction* MLP (`run_mlp_cifar32.py`) — need to confirm whether it's coordinate-conditioned (position -> pixel) or patch-conditioned; may need a variant |
| MPS-only model | — | Not found as a standalone baseline; the MPS feature extraction (`extract_mps_features`) is used everywhere as an input to other models, but no "MPS features -> linear/shallow readout, no quantum circuit" baseline exists on its own |
| Fourier-feature MLP | — | Not found. Sinusoidal positional encoding exists (`sinusoidal_positional_encoding`) but isn't packaged as a standalone Fourier-feature-MLP baseline |
| Classical quadratic/polynomial features | — | Not found as a dedicated baseline (closest relative: `real_newhvk_features` in `run_newhvk_suite.py`, a hand-crafted pairwise-product feature map, but that's framed as an HVK surrogate, not a baseline) |
| Parameter-matched tensor network | — | Not found |
| (extra, already exists) CNN | `cnn/` | Not in reviewer's list but already present |
| (extra, already exists) GAN | `gan/` | Not in reviewer's list but already present |
| (extra, already exists) PHL (Parameterized Hamiltonian Learning) | `phl/` | Classical analog of the Hamiltonian objective — directly relevant to `../04_hamiltonian_objective/` too |

So roughly half the reviewer's list already exists; the real gap is: MPS-only, Fourier-
feature MLP, polynomial features, and matched tensor-network.

## Status

Not started on the missing four. Existing baselines should be re-run against whichever
split protocol `../02_dataset_level_generalization/` settles on, for a fair, consistent
comparison table — sequence this after that item lands, not before, to avoid comparing
apples (150/50 dataset-level split) to oranges (old per-image or 6/4 held-out protocol).

## Next step

Once `../02_dataset_level_generalization/`'s run completes and its split protocol is
finalized, build the four missing baselines using the *same* train/held-out split and
same evaluation code, then assemble one comparison table (existing five architectures +
four new ones + HVK2D quantum + linear control) for the paper.
