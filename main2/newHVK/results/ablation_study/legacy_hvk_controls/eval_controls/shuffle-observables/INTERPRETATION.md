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

## Conclusion (as originally written, 2026-07 or earlier)

This legacy Exp-1 result should be treated as weak or negative evidence for
observable-position load-bearing behavior. It must not be cited as a 12.5 dB
shuffle degradation.

## 2026-07-29 update: independent re-verification disagrees sharply

An independent rebuild of the verifier
(`Main2/newHVK/verify_shuffle_permutations.py`, since the original script was
never committed — see `TODO/todo.md` B2) reproduces the exact safeguards this
document's own "Do this" list called for (permutation confirmed non-identity;
permuted tensor confirmed to be the exact object reaching the decoder) but
gets a **much larger** drop than documented here, and gets it **twice,
independently, under two different architecture variants**:

| Checkpoint | Baseline PSNR | Mean shuffle drop (5 perms) |
|---|---:|---:|
| Current default (`use_energy_feature=True`) | 33.41 dB | 15.70 +/- 0.79 dB |
| Legacy-equivalent (`use_energy_feature=False`, like-for-like with this document's setup) | 33.45 dB | **16.18 +/- 0.51 dB** |

Both numbers are far closer in magnitude to the *stale, discredited* ~12.5 dB
figure this document explicitly says must not be cited than to the 0.301 dB
figure documented above. Full per-permutation data:
`Main2/newHVK/results/verify_shuffle_permutations/verify_shuffle_permutations_result.json`
and `..._result_no_energy_feature.json`.

**This is not yet resolved, and this document's original 0.301 dB number is
not being silently overwritten.** Two possibilities, not distinguished yet:
(a) the original verification script (never committed, unrecoverable) had an
undetected bug in one of the exact two things it claimed to check (identity
permutation, decoder receiving a discarded copy) — the same failure mode
Task 1 in `TODO/todo.md` was written to guard against; or (b) some other
difference between the original checkpoint/protocol and this rebuild's
(different training script entirely, since the original is unrecoverable)
explains the gap. Do not cite either the 0.301 dB or the ~16 dB number as
settled until this is investigated further.
