# Baseline Algorithms

This folder contains independent baselines for comparing against the 1D HVK
pipeline in `Main/main.py` and the 2D HVK pipeline in `Main2/main.py`.

## Parameterized Hamiltonian Learning segmentation

`phl_segmentation/run_phl.py` implements a differentiable Ising-style
Parameterized Hamiltonian Learning baseline for binary image segmentation. The
learned Hamiltonian has unary intensity terms and horizontal/vertical pairwise
couplings. If a ground-truth mask is supplied, it trains against that mask. If no
mask is supplied, it uses an Otsu pseudo-mask so the script can run on the
current `monalisa.jpg` experiment.

Run:

```bash
python Baselines/phl_segmentation/run_phl.py
```

Useful options:

```bash
python Baselines/phl_segmentation/run_phl.py \
  --image-path Main/data/monalisa.jpg \
  --mask-path path/to/mask.png \
  --epochs 200
```

Outputs are written to `Baselines/outputs/phl/`:

- `phl_mask.png`
- `phl_probability.png`
- `phl_target_mask.png`
- `phl_training_history.csv`
- `phl_summary.json`

## GAN reconstruction

`gan_reconstruction/run_gan.py` implements a patch autoencoder generator with a
small convolutional discriminator. It is meant to benchmark image reconstruction
against HVK using MSE, PSNR, and SSIM. On larger datasets, add LPIPS/FID outside
this minimal script.

Run:

```bash
python Baselines/gan_reconstruction/run_gan.py
```

Useful options:

```bash
python Baselines/gan_reconstruction/run_gan.py \
  --image-path Main/data/monalisa.jpg \
  --epochs 200 \
  --patch-size 64
```

Outputs are written to `Baselines/outputs/gan/`:

- `gan_reconstruction.png`
- `gan_target.png`
- `gan_training_history.csv`
- `gan_summary.json`

## Interpretation

PHL is a segmentation baseline: its natural metrics are Dice, IoU, boundary
F-score, and pixel accuracy. GAN is a reconstruction/perceptual baseline: its
natural metrics are MSE, PSNR, SSIM, LPIPS, and FID. HVK remains different from
both because its Hamiltonian term regularizes a quantum-inspired latent
autoencoding path and also produces energy/order-parameter diagnostics.
