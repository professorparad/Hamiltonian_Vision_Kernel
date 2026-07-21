# HVK — Consolidated Results Summary

> **Historical snapshot — superseded for submission.** This document records an
> earlier internal review and includes exploratory aggregates that are not
> retained as manuscript evidence. In particular, its Monalisa component
> attribution lacks reproducible per-seed artifacts, and its leakage verdict
> predates the corrected restricted diagnostic. For submission claims and
> current statistics, use `latex_outputs/paper_latex/paper_hvk.tex`,
> `latex_outputs/paper_latex/supplementary_study.tex`, and their artifact map.

> **Historical snapshot — superseded for submission.** This document records an
> earlier internal review and includes exploratory aggregates that are not
> retained as manuscript evidence. In particular, its Monalisa component
> attribution lacks reproducible per-seed artifacts, and its leakage verdict
> predates the corrected restricted diagnostic. For submission claims and
> current statistics, use `latex_outputs/paper_latex/paper_hvk.tex`,
> `latex_outputs/paper_latex/supplementary_study.tex`, and their artifact map.

Companion to `algorithm.md` (implementation), `report.md` (experiment spec), and
`report_2.md` (critical review). This file collects **all numeric results** across
the ablation study, the multi-image benchmarks, and the second-generation newHVK
suite, with an honest bottom line.

All PSNR in dB; higher is better. SSIM ∈ [0,1]; higher is better. MSE lower is better.

---

## 1. Headline verdict

- HVK **reconstructs a single image well** (~32 dB) but this is a **per-image
  optimization**, not a learned representation (zero-shot to a new image = 7.8 dB,
  below noise floor).
- **No quantum component is demonstrably load-bearing.** Freeze-quantum,
  no-entanglement, no-MPS, no-energy-loss, and a classical `tanh` replacement all
  **match or beat** the full model.
- On **multi-seed held-out CIFAR-10**, the entangling quantum model **ties or
  slightly trails** a matched classical model.
- The one "quantum advantage" number (newHVK nonlocal, R²≈1.0) is **circular by
  construction** (label leakage) and carries no evidence.
- **Publishable framing:** honest ablation / negative-result study, or a pivot to
  a genuinely nonlocal task with leakage-free controls. **Not** a quantum-advantage claim.

---

## 2. Core ablation suite (single image, seed 42)

Source: `experiments/quantum_contribution/results/`. Baseline = full HVK1D, 32.24 dB.

| Experiment | PSNR | SSIM | vs baseline | Reading |
|---|---:|---:|---:|---|
| **Baseline (full HVK)** | 32.24 | 0.992 | — | reference |
| Classical `Linear+tanh` replacement | **33.45** | 0.994 | +1.21 | VQC beaten by trivial classical map |
| No energy loss | 33.30 | 0.994 | +1.06 | Hamiltonian term inert |
| Classical-matched | 33.19 | 0.994 | +0.95 | same |
| **Freeze quantum** (VQC+J fixed) | **33.11** | 0.993 | +0.87 | trained quantum params not needed |
| No MPS (patch stats) | 32.69 | 0.993 | +0.45 | MPS encoder adds nothing |
| No entanglement | 32.62 | 0.993 | +0.38 | entanglement adds nothing |
| ZZ-only observables | 32.88 | 0.993 | +0.64 | XX/YY not needed |
| Random VQC (untrained) | 25.98 | 0.966 | −6.26 | needs *structure*, not *quantum* |
| **Freeze classical** (decoder frozen) | 10.93 | 0.021 | −21.3 | **all learning is classical** |

**Only load-bearing components:** the classical decoder/projections, and having
*some* structured latent. No isolated quantum contribution.

### Eval controls

| Control | PSNR | Note |
|---|---:|---|
| Zero observables + real positions | 14.85 | position alone insufficient ✅ |
| Random latent + real positions | 12.14 | noise floor |
| **Shuffle observables, verified** | **32.04** | five non-identity permutations give only −0.301 ± 0.054 dB; weak/negative load-bearing evidence |

### Bond-dim / qubit sweeps (all ≈ baseline, non-monotonic → within noise)

χ1 32.95 · χ2 32.75 · χ4 32.24 · χ8 32.51 | q4 32.71 · q6 32.24 · q8 33.07

### Step sweep (convergence)

30 → 18.30 · 60 → 23.12 · 120 → 28.75 · 240 → 32.55 · 500 → 33.38 dB.
**120 steps is underfit;** use ≥240 for fair comparisons.

---

## 3. Generalization (the most damaging result)

| Run | PSNR | SSIM | Reading |
|---|---:|---:|---|
| Second image (zero-shot, no retrain) | **7.78** | 0.020 | **memorized one image** (below ~12 dB noise floor) |
| Multi-image (trained across images) | 28.31 | 0.986 | only viable path to a real representation |

---

## 4. Multi-image benchmarks vs classical baselines

Source: `main2/newHVK/results/baselines/`.

### CIFAR-10, 32×32, 5 images (HVK is competitive here)

| Model | PSNR | SSIM | Class |
|---|---:|---:|---|
| MLP (overfit per-image) | 75.14 | ~1.000 | classical (memorizes) |
| CNN | 40.61 | 0.999 | classical |
| **HVK2D** | **34.50** | 0.996 | quantum (best HVK) |
| SymmetricHVK1D | 32.81 | 0.994 | quantum |
| HVK1D | 32.42 | 0.993 | quantum |
| GAN | 26.03 | 0.969 | classical |
| Autoencoder | 20.78 | 0.882 | classical |
| PHL | 14.01 | 0.788 | classical |

*HVK2D > HVK1D > Sym here; HVK beats AE/GAN but the CNN and (memorizing) MLP win.*

### Mona Lisa, 256×256, single image (HVK collapses at large size)

| Model | PSNR | SSIM |
|---|---:|---:|
| MLP | 70.92 | ~1.000 |
| CNN | 43.65 | 0.999 |
| GAN | 27.06 | 0.970 |
| SymmetricHVK1D | 19.41 | 0.780 |
| HVK1D | 19.34 | 0.778 |
| HVK2D | 19.26 | 0.776 |
| Autoencoder | 18.88 | 0.760 |

*At 256×256 all HVK variants drop to ~19 dB — the 6-qubit / 27-observable
bottleneck does not scale to large patches.*

---

## 5. newHVK second-generation suite (multi-seed, stricter)

Source: `main2/newHVK/results/`.

### 5a. Real held-out CIFAR-10 — 20 images, multi-seed (TRUSTWORTHY, negative)

| Model | PSNR | SSIM |
|---|---:|---:|
| local-observables-only / raw-linear-classical | **20.66** | 0.862 |
| zz-only | 20.52 | 0.858 |
| no-entanglement | 20.24 | 0.851 |
| shuffled-pair-observables | 20.17 | 0.848 |
| **newHVK entangling (real CIFAR)** | 20.07 | 0.846 |
| strict-classical-rff | 17.06 | 0.706 |
| random-vqc | 10.99 | 0.016 |

**Entangling quantum model ties/trails a plain local-linear classical model.** No advantage.

### 5b. Full ablation, 5 seeds (mean PSNR ± std)

| Model | PSNR | ±std | R² |
|---|---:|---:|---:|
| newHVK entangling | 32.77 | 5.52 | 0.974 |
| freeze-quantum | 27.34 | 8.66 | 0.853 |
| no-entanglement | 15.03 | 0.24 | 0.019 |
| raw-linear-classical | 15.00 | 0.26 | 0.013 |
| parameter-matched-classical | 14.82 | 0.21 | −0.03 |
| random-vqc | 14.25 | 0.20 | −0.17 |
| freeze-classical | 12.17 | 0.45 | −0.90 |

*(Large std on the top rows ⇒ differences not tightly separated. This is a
proxy pair-correlation task, not real-image reconstruction — read with 5a.)*

### 5c. Shot-noise sweep (finite-shot robustness)

128→8192 shots: PSNR flat at **≈18.4–18.6 dB**, SSIM ≈0.79. Reconstruction is
**shot-count insensitive** above ~256 shots — good NISQ robustness signal.

### 5d. IBM hardware probe (feasibility, 6 qubits, transpiled)

| Variant | Depth | CX count |
|---|---:|---:|
| hvk1d (chain) | 18 | 10 |
| hvk2d (grid) | 18 | 14 |

*Circuits are small and hardware-runnable; grid uses more two-qubit gates.*

### 5e. CIFAR "nonlocal advantage" — ⚠️ INVALID (label leakage)

Reported: entangling R²=0.99999999999997, PSNR=120 dB, MSE≈4e-15 vs controls
R²≈0.2. **Circular:** the regression target is built from the exact pair-product
features handed only to the entangling model (target `run_newhvk_suite.py:1374-1385`,
features `1396-1413`).
Any model given those columns scores R²=1.0. **Do not cite as advantage.**

---

## 6. What is and isn't established

**Established:**
- Classical decoder/projections are necessary and sufficient (freeze-classical fails).
- A structured latent is required (random-VQC poor), but need not be quantum.
- Positions alone are insufficient (zero-latent control).
- HVK is competitive at 32×32 but does not scale to 256×256; classical CNN/MLP win throughout.
- Circuits are hardware-feasible and shot-noise robust at small scale.

**NOT established (per report_2):**
- That trained quantum params, entanglement, the MPS encoder, or the energy term contribute.
- Any quantum advantage over matched classical controls on real held-out images.
- Generalization beyond the training image.

---

## 7. Required actions before publication

1. **Resolved:** Exp 1 shuffle was re-verified; the true effect is small, so do not claim a large shuffle degradation.
2. **Re-run core ablations at ≥240 steps** (120 is underfit) with 5 seeds + error bars.
3. **Delete or redesign the nonlocal-advantage diagnostic** (remove target–feature leakage; give all models equal access to nonlocal terms).
4. **Adopt the honest framing:** ablation/negative-result paper built on the real held-out CIFAR result, or pivot to a leakage-free nonlocal task.
5. Drop the "phase transition" narrative until it survives a control (it currently fires even with the Hamiltonian off).
