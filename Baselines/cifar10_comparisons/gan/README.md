# GAN Baseline

This folder contains a small patch-level GAN-style reconstruction baseline for
32x32 grayscale CIFAR images.

It is not a large generative model. The generator behaves like a patch
autoencoder, while the discriminator gives an adversarial signal on 8x8 patches.
The generator loss combines:

- reconstruction MSE
- adversarial binary cross-entropy

This gives a slightly different comparison point from plain MSE autoencoders.

## Run

```powershell
python Baselines\cifar10_comparisons\gan\run_gan_cifar32.py --count 5 --epochs 200 --device cuda
```

Through the combined runner:

```powershell
python Baselines\cifar10_comparisons\main.py --methods gan --count 5 --epochs 200 --device cuda
```

## Outputs

The script writes:

- `outputs/gan_cifar32_metrics.csv`
