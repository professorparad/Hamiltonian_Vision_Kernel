# Exp 1 Interpretation — Shuffled Observables

## What Was Tested

The model was trained normally. At evaluation time, the observable vectors were
shuffled across patches while the positional encodings stayed in the correct
patch locations.

This tests whether the decoder is using the quantum observable vector for
patch-specific content, or whether it is mostly reconstructing from position.

## Result

| Reconstruction | MSE vs Original | PSNR vs Original |
|---|---:|---:|
| Normal HVK reconstruction | 0.0006018857 | 32.20 dB |
| Shuffled-observable reconstruction | 0.0107267685 | 19.70 dB |

Additional comparison:

```text
MSE(shuffled, normal) = 0.0099088643
```

## Meaning

The shuffled reconstruction is much worse than the normal reconstruction.

That means the decoder is not reconstructing the image from position alone. The
observable vectors carry patch-specific information, and disrupting the mapping
between observables and patch positions strongly degrades reconstruction.

## Conclusion

For this run, HVK observables are load-bearing. The quantum latent features are
contributing useful patch-specific content to the reconstruction pipeline.

