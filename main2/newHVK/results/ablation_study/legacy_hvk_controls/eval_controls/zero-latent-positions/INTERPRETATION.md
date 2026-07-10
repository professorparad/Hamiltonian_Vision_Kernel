# Exp 2 Interpretation — Zero Quantum Features

## What Was Tested

The model was trained normally. At evaluation time, the decoder was given zero
observable vectors while the positional encodings stayed correctly assigned to
their patch locations.

This tests whether the decoder can reconstruct the image from position alone,
or whether the quantum observable vector is needed for patch-specific content.

## Result

| Reconstruction | MSE vs Original | PSNR vs Original |
|---|---:|---:|
| Normal HVK reconstruction | 0.0006213221 | 32.07 dB |
| Zero-observable reconstruction | 0.0294627398 | 15.31 dB |
| Random-latent reconstruction | 0.0715787932 | 11.45 dB |

Additional comparison:

```text
MSE(zero-observable, normal) = 0.0303193424
```

## Meaning

The zero-observable reconstruction is much worse than the normal
reconstruction.

That means the decoder is not reconstructing the image from position alone.
Real positional encodings are not enough; the observable vectors carry important
patch-specific information.

## Conclusion

For this run, HVK observables are load-bearing. Removing the quantum observable
features causes a large degradation in reconstruction quality.

