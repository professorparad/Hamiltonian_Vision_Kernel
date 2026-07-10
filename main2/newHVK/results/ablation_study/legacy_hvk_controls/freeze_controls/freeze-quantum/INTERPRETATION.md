# Exp 4 Interpretation - Freeze Quantum

## What Was Tested

The model was trained in `freeze-quantum` ablation mode:

- VQC weights frozen
- Hamiltonian couplings `Jx`, `Jy`, `Jz` frozen
- feature projection, position projection, and decoder train normally

This tests whether the classical trainable stack can reconstruct the image when
the quantum circuit is only a fixed random feature map.

## Result

| Metric | Value |
|---|---:|
| Final reconstruction MSE | 0.0004881928 |
| Final PSNR | 33.11 dB |
| Final SSIM | 0.9934 |
| Final total loss | -0.0005387817 |
| Final reconstruction loss | 0.0000511228 |
| Final energy loss | -0.0589904487 |
| Mean energy | -0.0592251644 |

Additional controls from `comparison_metrics`:

| Output | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Freeze-quantum reconstruction | 0.0004881928 | 33.11 dB | 0.9934 |
| MPS baseline | 0.0021553168 | 26.66 dB | 0.9703 |
| Random latent | 0.0547767580 | 12.61 dB | 0.3501 |
| Zero latent | 0.0426607132 | 13.70 dB | 0.0837 |

Phase transition summary:

| Quantity | Value |
|---|---:|
| Phase transition detected | yes |
| Critical epoch | 10 |
| Max susceptibility | 0.0079734474 |
| Susceptibility threshold | 0.0030427969 |
| Order-parameter jump | 0.0079734474 |

## Files

- Metrics: [metrics.json](metrics.json)
- Reconstruction plot: [reconstructions.png](reconstructions.png)
- Training curves: [training_curves.png](training_curves.png)
- Order parameter curve: [hvk_order_parameter_curve.png](hvk_order_parameter_curve.png)
- Epoch reconstruction table: [hvk_epoch_reconstruction_table.csv](hvk_epoch_reconstruction_table.csv)

## Interpretation

Freeze-quantum reconstructs very well. The final PSNR is 33.11 dB and SSIM is
0.9934, which is comparable to or better than the shared baseline run.

This means trained quantum weights are not required for strong reconstruction
on this single-image task. The classical projections and decoder can learn a
high-quality reconstruction using fixed random VQC features.

The random-latent and zero-latent controls are much worse, so the decoder still
needs a structured input signal. However, this experiment shows that the
structure does not need to come from trained quantum parameters.

## Conclusion

Exp 4 weakens the claim that trained quantum parameters are load-bearing. The
classical trainable stack can reconstruct strongly even when the VQC and
Hamiltonian couplings are frozen.

