# Quantum Contribution Experiments — Consolidated Interpretation

This file is the central interpretation log for the quantum-contribution ablation
study. Each experiment should add its result links, comparison metrics, and final
interpretation here after it is run.

## Current Bottom Line

Experiments 1, 2, 3, 4, and 7 are complete. Together, they give a more nuanced
picture than the initial sanity checks alone:

- Shuffling observables across patch positions causes a large reconstruction
  drop.
- Replacing observables with zeros while keeping real positions also causes a
  large reconstruction drop.
- Therefore, the decoder is not reconstructing from position alone. The
  observable vector carries patch-specific information.
- Freezing the classical decoder/projections makes reconstruction fail, so the
  classical trainable stack is necessary.
- Freezing the quantum parameters still reconstructs well, so trained quantum
  weights are not necessary for this single-image task.
- Replacing the VQC with a classical `Linear(6 -> 27) + tanh` map also
  reconstructs well, slightly beating the current VQC baseline.

Current best interpretation: HVK observables are useful latent signals, but the
completed ablations do not support a strong claim that trained quantum
parameters or the VQC specifically are responsible for reconstruction quality.
The classical decoder/projection stack appears to carry most of the learning
burden in the completed runs.

**Important control caveat:** the currently recorded Exp 1 and Exp 2 numbers
were produced by separate training runs, so their "Normal HVK" rows are close
but not identical. They are valid within-experiment controls, but the Exp 1 vs
Exp 2 baseline rows should not be treated as a shared-checkpoint comparison.
The fixed `run_eval_controls.sh` now trains one `shared-baseline-seed-42`
checkpoint and evaluates both controls from that exact checkpoint.

**Metric policy:** all new experiment runs must report MSE, PSNR, and SSIM in
`metrics.json`. For eval controls, report those metrics for the normal
reconstruction and for each perturbation/control image.

## Result Index

| Exp | Name | Status | Result folder | Primary summary | Interpretation |
|---:|---|---|---|---|---|
| 1 | Shuffle observables at eval | Done | [shuffle-observables](eval_controls/shuffle-observables) | [shuffle_eval_summary.json](eval_controls/shuffle-observables/shuffle_eval_summary.json) | [local note](eval_controls/shuffle-observables/INTERPRETATION.md) |
| 2 | Zero observables + real positions | Done | [zero-latent-positions](eval_controls/zero-latent-positions) | [zero_latent_eval_summary.json](eval_controls/zero-latent-positions/zero_latent_eval_summary.json) | [local note](eval_controls/zero-latent-positions/INTERPRETATION.md) |
| 3 | Freeze classical, train quantum only | Done | [freeze-classical](freeze_controls/freeze-classical) | [metrics.json](freeze_controls/freeze-classical/metrics.json) | [local note](freeze_controls/freeze-classical/INTERPRETATION.md) |
| 4 | Freeze quantum, train classical only | Done | [freeze-quantum](freeze_controls/freeze-quantum) | [metrics.json](freeze_controls/freeze-quantum/metrics.json) | [local note](freeze_controls/freeze-quantum/INTERPRETATION.md) |
| 5 | Random VQC output | Pending | [random-vqc](vqc_replacements/random-vqc) | pending | pending |
| 6 | Generalization to second image | Pending | [generalization](generalization) | pending | pending |
| 7 | Classical tanh replacement | Done | [classical-replacement](vqc_replacements/classical-replacement) | [metrics.json](vqc_replacements/classical-replacement/metrics.json) | [local note](vqc_replacements/classical-replacement/INTERPRETATION.md) |
| 8 | No MPS, patch statistics only | Pending | [no-mps](feature_bottleneck/no-mps) | pending | pending |
| 9 | No entanglement | Pending | [no-entanglement](vqc_replacements/no-entanglement) | pending | pending |
| 10 | Step count sweep | Pending | [step_sweep](step_sweep) | pending | pending |
| 11 | No energy loss | Pending | [no-energy-loss](hamiltonian_controls/no-energy-loss) | pending | pending |
| 12 | No observable noise | Pending | [no-obs-noise](hamiltonian_controls/no-obs-noise) | pending | pending |

## Completed Experiments

### Exp 1 — Shuffle Observables at Eval

**Question:** Does the decoder use observable values, or can it reconstruct from
position alone?

**Method:** Train normally. At evaluation time, randomly permute observable
vectors across patches while keeping positional encodings fixed.

**Files:**

- Result folder: [eval_controls/shuffle-observables](eval_controls/shuffle-observables)
- Summary JSON: [shuffle_eval_summary.json](eval_controls/shuffle-observables/shuffle_eval_summary.json)
- Metrics JSON: [metrics.json](eval_controls/shuffle-observables/metrics.json)
- Reconstruction plot: [reconstructions.png](eval_controls/shuffle-observables/standard_hvk1d/reconstructions.png)
- Training curve: [training_curves.png](eval_controls/shuffle-observables/standard_hvk1d/training_curves.png)
- Order parameter curve: [hvk_order_parameter_curve.png](eval_controls/shuffle-observables/standard_hvk1d/hvk_order_parameter_curve.png)
- Epoch reconstruction table: [hvk_epoch_reconstruction_table.csv](eval_controls/shuffle-observables/standard_hvk1d/hvk_epoch_reconstruction_table.csv)
- Existing local interpretation: [INTERPRETATION.md](eval_controls/shuffle-observables/INTERPRETATION.md)

| Reconstruction | MSE vs Original | PSNR vs Original | SSIM vs Original |
|---|---:|---:|---:|
| Normal HVK | 0.0006018857 | 32.20 dB | 0.9928 |
| Shuffled observables | 0.0107267685 | 19.70 dB | 0.8423 |

| Comparison | Value |
|---|---:|
| MSE(shuffled, normal) | 0.0099088643 |
| MSE degradation vs normal | 17.82x worse |
| PSNR drop | 12.51 dB |

**Interpretation:** Shuffling observable vectors destroys the correct
observable-position pairing. The reconstruction gets much worse, so the decoder
is not ignoring observables. The observables encode patch-specific content.

**Metric note:** SSIM was backfilled from the saved reconstruction arrays for
this completed run. Future reruns will write SSIM directly into `metrics.json`.

**Conclusion:** Exp 1 supports the claim that HVK observables are load-bearing.

### Exp 2 — Zero Observables with Real Positions

**Question:** Can the decoder reconstruct the image from positional encoding
alone?

**Method:** Train normally. At evaluation time, replace observable vectors with
zeros while preserving the real positional encodings. Also compare to random
latent vectors.

**Files:**

- Result folder: [eval_controls/zero-latent-positions](eval_controls/zero-latent-positions)
- Summary JSON: [zero_latent_eval_summary.json](eval_controls/zero-latent-positions/zero_latent_eval_summary.json)
- Metrics JSON: [metrics.json](eval_controls/zero-latent-positions/metrics.json)
- Reconstruction plot: [reconstructions.png](eval_controls/zero-latent-positions/reconstructions.png)
- Training curve: [training_curves.png](eval_controls/zero-latent-positions/training_curves.png)
- Order parameter curve: [hvk_order_parameter_curve.png](eval_controls/zero-latent-positions/hvk_order_parameter_curve.png)
- Epoch reconstruction table: [hvk_epoch_reconstruction_table.csv](eval_controls/zero-latent-positions/hvk_epoch_reconstruction_table.csv)
- Existing local interpretation: [INTERPRETATION.md](eval_controls/zero-latent-positions/INTERPRETATION.md)

| Reconstruction | MSE vs Original | PSNR vs Original | SSIM vs Original |
|---|---:|---:|---:|
| Normal HVK | 0.0006213221 | 32.07 dB | 0.9929 |
| Zero observables + real positions | 0.0294627398 | 15.31 dB | 0.8337 |
| Random latent + real positions | 0.0715787932 | 11.45 dB | 0.1217 |

| Comparison | Value |
|---|---:|
| MSE(zero, normal) | 0.0303193424 |
| MSE degradation vs normal | 47.42x worse |
| PSNR drop, zero vs normal | 16.76 dB |
| PSNR gap, zero vs random latent | 3.86 dB |

**Interpretation:** Zeroing the observable vector while keeping real patch
positions causes a severe quality collapse. The decoder cannot reconstruct the
image from position alone. The random-latent result is worse still, which gives a
rough noise-floor comparison.

**Metric note:** SSIM was backfilled from the saved reconstruction arrays for
this completed run. Future reruns will write SSIM directly into `metrics.json`.

**Conclusion:** Exp 2 rules out simple position memorization for this run.

## Cross-Experiment Comparison

The table below compares each perturbation against the normal reconstruction
from the same experiment run. Because Exp 1 and Exp 2 were originally run
separately, do not compare their normal rows as if they came from one shared
checkpoint.

| Experiment | Normal MSE | Perturbed MSE | MSE Ratio | Normal PSNR | Perturbed PSNR | PSNR Drop | Perturbed SSIM | Meaning |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Exp 1: shuffled observables | 0.0006018857 | 0.0107267685 | 17.82x | 32.20 dB | 19.70 dB | 12.51 dB | 0.8423 | Observable-position pairing matters |
| Exp 2: zero observables | 0.0006213221 | 0.0294627398 | 47.42x | 32.07 dB | 15.31 dB | 16.76 dB | 0.8337 | Positions alone are insufficient |
| Exp 2: random latent | 0.0006213221 | 0.0715787932 | 115.20x | 32.07 dB | 11.45 dB | 20.61 dB | 0.1217 | Decoder output is near noise floor without real observables |

## What These Results Do and Do Not Prove

### Supported So Far

- The decoder uses observable values.
- The observable vector contains patch-specific information.
- Correct observable-to-position assignment matters.
- Real positions alone are not enough for high-quality reconstruction.

### Not Yet Proven

- That trained quantum parameters are necessary.
- That the VQC is better than fixed random quantum features.
- That the VQC is better than random noise of the same dimension.
- That the VQC is better than a classical tanh replacement.
- That entanglement specifically contributes.
- That the MPS feature extractor is necessary.
- That the energy loss contributes useful regularization.
- That the learned model generalizes to a second image.

## Pending Experiment Interpretation Template

Copy this block when adding a new completed experiment.

```markdown
### Exp N — Experiment Name

**Question:** ...

**Method:** ...

**Files:**

- Result folder: [...]
- Metrics JSON: [...]
- Reconstruction plot: [...]
- Training curve: [...]

| Condition | MSE | PSNR | SSIM | Notes |
|---|---:|---:|---:|---|
| Baseline | ... | ... | ... | ... |
| Ablation | ... | ... | ... | ... |

**Interpretation:** ...

**Conclusion:** ...
```

## Recommended Reading Order

1. Read Exp 2 first. It answers whether the decoder can reconstruct from
   position alone.
2. Read Exp 1 next. It answers whether observables must be assigned to the
   correct patches.
3. Run/read Exp 4 next. It tests whether trained quantum weights matter, or
   whether fixed random VQC features are enough.
4. Run/read Exp 5 and Exp 7 after that. They test whether structured quantum
   outputs beat random noise and a classical tanh replacement.
5. Run/read Exp 9. It isolates whether entanglement specifically helps.
6. Run/read Exp 11. It tests whether the Hamiltonian energy term is useful.
7. Run/read Exp 8 and Exp 6 last. They test the tensor-network encoder and
   generalization beyond the single training image.
