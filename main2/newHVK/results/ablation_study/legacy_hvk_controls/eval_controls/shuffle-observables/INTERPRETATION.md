# Exp 1 Interpretation — Shuffled Observables

## What Was Tested

The model was trained normally. At evaluation time, the observable vectors were
permuted across patches while the positional encodings stayed in their original
patch locations.

## Verified Result

The current saved `shuffle_eval_summary.json` supports only a small degradation:

| Reconstruction | MSE vs Original | PSNR vs Original | SSIM vs Original |
|---|---:|---:|---:|
| Normal HVK reconstruction | 0.0005977100 | 32.24 dB | 0.9919 |
| Shuffled-observable reconstruction | 0.0006248252 | 32.04 dB | 0.9916 |

The single-run PSNR drop is 0.19 dB, with `MSE(shuffled, normal) =
2.2958744e-05`.

A repeated post-training verification over five non-identity permutations gives
a mean PSNR drop of `0.301 +/- 0.054 dB` with range `0.236` to `0.366` dB.

## Conclusion

This legacy Exp-1 result should be treated as weak or negative evidence for
observable-position load-bearing behavior. It must not be cited as a 12.5 dB
shuffle degradation.
