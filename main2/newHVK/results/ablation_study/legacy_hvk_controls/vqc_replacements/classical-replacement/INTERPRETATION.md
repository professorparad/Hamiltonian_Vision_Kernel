# Exp 7 Interpretation - Classical Tanh VQC Replacement

## What Was Tested

The quantum circuit was replaced by a classical `Linear(6 -> 27) + tanh` map.
The energy term is inactive in this mode because there are no Heisenberg
couplings or measured XX/YY/ZZ slices from a VQC.

This tests whether the VQC provides reconstruction value beyond a simple
classical nonlinear map with the same input and output shape.

## Result

| Metric | Value |
|---|---:|
| Final reconstruction MSE | 0.0004514573 |
| Final PSNR | 33.45 dB |
| Final SSIM | 0.9939 |
| Final total loss | 0.0000087335 |
| Final reconstruction loss | 0.0000087335 |
| Final energy loss | 0.0000000000 |
| Mean energy | 0.0000000000 |

Additional controls from `comparison_metrics`:

| Output | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Classical replacement reconstruction | 0.0004514573 | 33.45 dB | 0.9939 |
| MPS baseline | 0.0021553168 | 26.66 dB | 0.9703 |
| Random latent | 0.0324836075 | 14.88 dB | 0.4284 |
| Zero latent | 0.0550898537 | 12.59 dB | 0.0360 |

Phase transition summary:

| Quantity | Value |
|---|---:|
| Phase transition detected | yes |
| Critical epoch | 11 |
| Max susceptibility | 0.0066641495 |
| Susceptibility threshold | 0.0023720281 |
| Order-parameter jump | 0.0066641495 |

## Files

- Metrics: [metrics.json](metrics.json)
- Reconstruction plot: [reconstructions.png](reconstructions.png)
- Training curves: [training_curves.png](training_curves.png)
- Order parameter curve: [hvk_order_parameter_curve.png](hvk_order_parameter_curve.png)
- Epoch reconstruction table: [hvk_epoch_reconstruction_table.csv](hvk_epoch_reconstruction_table.csv)

## Interpretation

The classical tanh replacement reconstructs extremely well. It slightly exceeds
the shared baseline run and the freeze-quantum run on MSE, PSNR, and SSIM.

This is a serious constraint on the strong quantum claim. In this run, the VQC
is not necessary to achieve high-quality reconstruction. A simple classical
nonlinear map with the same latent output size is enough.

The result does not prove that the VQC is useless in every setting, but it does
show that this single-image reconstruction benchmark is not sufficient evidence
for a quantum representational advantage.

## Conclusion

Exp 7 weakens the strong quantum-advantage claim. The classical replacement
matches or slightly beats the VQC baseline, so the paper should not claim that
the quantum circuit is uniquely responsible for reconstruction quality on this
task without further evidence.

