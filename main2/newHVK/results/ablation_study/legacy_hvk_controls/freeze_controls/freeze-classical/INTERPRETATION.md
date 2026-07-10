# Exp 3 Interpretation - Freeze Classical

## What Was Tested

The model was trained in `freeze-classical` ablation mode:

- decoder frozen
- feature projection frozen
- position projection frozen
- only quantum/Hamiltonian parameters trainable

This tests whether the small quantum parameter set can learn a useful image
reconstruction while the large classical decoder/projection stack remains at
random initialization.

## Result

| Metric | Value |
|---|---:|
| Final reconstruction MSE | 0.0806391537 |
| Final PSNR | 10.93 dB |
| Final SSIM | 0.0214 |
| Final total loss | 0.0441795886 |
| Final reconstruction loss | 0.0810948014 |
| Final energy loss | -3.6915214062 |
| Mean energy | -3.7125227451 |

Phase transition summary:

| Quantity | Value |
|---|---:|
| Phase transition detected | yes |
| Critical epoch | 52 |
| Max susceptibility | 0.0023872554 |
| Susceptibility threshold | 0.0020826500 |
| Order-parameter jump | 0.0023872554 |

## Files

- Metrics: [metrics.json](metrics.json)
- Reconstruction plot: [reconstructions.png](reconstructions.png)
- Training curves: [training_curves.png](training_curves.png)
- Order parameter curve: [hvk_order_parameter_curve.png](hvk_order_parameter_curve.png)
- Epoch reconstruction table: [hvk_epoch_reconstruction_table.csv](hvk_epoch_reconstruction_table.csv)

## Interpretation

Freeze-classical fails to reconstruct the image. The final PSNR is only
10.93 dB and SSIM is 0.0214, which is essentially a noise-level reconstruction.

This result is expected: the decoder is a large randomly initialized network,
and freezing it prevents it from learning the image-to-patch decoding map. The
quantum parameters alone cannot compensate for a frozen random decoder.

The detected phase-transition-like event should not be interpreted as evidence
of useful reconstruction by itself. The reconstruction metrics are poor, so the
order-parameter dynamics are not enough to make the model useful in this
condition.

## Conclusion

Exp 3 shows that quantum parameters alone are not sufficient to reconstruct the
image when the classical decoder/projections are frozen. The classical decoder
is necessary for reconstruction quality in this architecture.

