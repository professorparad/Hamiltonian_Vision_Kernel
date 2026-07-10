# Exp 7b Interpretation - Parameter-Matched Classical Replacement

## What Was Tested

The VQC was replaced by a parameter-matched classical map while keeping the rest
of the architecture unchanged:

```text
combined 6-dim input -> Linear(6 -> 1, no bias) -> Linear(1 -> 27, no bias) -> tanh
```

This replacement has `6 + 27 = 33` trainable parameters, which is close to the
VQC's 36 trainable circuit weights. The decoder, MPS features, positional path,
and 27-dimensional observable interface are unchanged.

## Result

| Metric | Value |
|---|---:|
| Final reconstruction MSE | 0.0004796785 |
| Final PSNR | 33.19 dB |
| Final SSIM | 0.9935 |
| Final total loss | 0.0000407477 |
| Final reconstruction loss | 0.0000407477 |
| Final energy loss | 0.0000000000 |
| Mean energy | 0.0000000000 |

Additional controls from `comparison_metrics`:

| Output | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Parameter-matched classical reconstruction | 0.0004796785 | 33.19 dB | 0.9935 |
| MPS baseline | 0.0021553168 | 26.66 dB | 0.9703 |
| Random latent | 0.0253409874 | 15.96 dB | 0.5391 |
| Zero latent | 0.0485375673 | 13.14 dB | 0.0622 |

Phase transition summary:

| Quantity | Value |
|---|---:|
| Phase transition detected | yes |
| Critical epoch | 12 |
| Max susceptibility | 0.0042236205 |
| Susceptibility threshold | 0.0017240538 |
| Order-parameter jump | 0.0042236205 |

## Files

- Metrics: [metrics.json](metrics.json)
- Reconstruction plot: [reconstructions.png](reconstructions.png)
- Training curves: [training_curves.png](training_curves.png)
- Order parameter curve: [hvk_order_parameter_curve.png](hvk_order_parameter_curve.png)
- Epoch reconstruction table: [hvk_epoch_reconstruction_table.csv](hvk_epoch_reconstruction_table.csv)
- Epoch correlation table: [hvk_epoch_correlation_table.csv](hvk_epoch_correlation_table.csv)

## Interpretation

The parameter-matched classical replacement reconstructs very well: 33.19 dB
PSNR and 0.9935 SSIM. It is slightly below the larger 189-parameter classical
replacement, but still above the shared VQC baseline.

This is the fairest classical replacement result so far because the parameter
count is close to the VQC circuit-weight count. Since this small classical map
matches or beats the VQC baseline, the completed evidence does not support a
claim that the VQC circuit itself is the key source of reconstruction advantage
on this task.

The result does not mean the circuit is useless in all settings. It means that
this single-image reconstruction benchmark has not isolated a quantum-specific
advantage.

## Conclusion

Exp 7b strongly weakens the current quantum-advantage claim. A 33-parameter
classical tanh map can match or exceed the VQC baseline while preserving the
same latent interface and outer architecture.

