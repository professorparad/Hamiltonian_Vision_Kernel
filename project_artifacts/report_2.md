# Quantum Contribution Ablation — Critical Review (report_2)

**Reviewer role:** publication-quality scrutiny of the results in
`experiments/quantum_contribution/results/`.
**Date:** 2026-07-10
**Verdict in one line:** The student ran an impressively complete ablation suite
(all 10 planned experiments plus extras), and it is thorough and well-organized —
**but the results, read honestly, do not support a quantum-advantage claim, and
one headline number (Exp 1) is internally contradictory and cannot be trusted as
reported.**

> **Addendum (2026-07-12): review of the new `main2/newHVK/` workspace.**
> Since the first review the student built a second-generation study
> (`main2/newHVK/`, commits `2cc8631`→`c4c2e2c`) that directly answers most of
> report_2's methodology critique: 5-seed error bars, real held-out CIFAR-10
> (20 images), same-width classical controls, shot-noise/hardware proxies, and
> an IBM circuit probe. This is a large, genuine improvement. **But its single
> "advantage" headline (the CIFAR nonlocal diagnostic, R²≈1.0 / PSNR=120 dB) is
> circular by construction and must not be published as an advantage.** See
> §9 below. The honest held-out CIFAR result (§9) is the one to build on.

---

## 1. Coverage — what was actually done (credit where due)

The student went far beyond the 5 core experiments. All of Exp 1–10 were run, plus
extra controls (random-VQC, ZZ-only observables, bond-dim sweep χ∈{1,2,4,8},
qubit-count sweep, contrastive, no-obs-noise, classical-matched). Every run reports
MSE, PSNR, **and SSIM** as requested, uses **seed 42**, and a **shared
baseline checkpoint** was created (`baseline/shared-seed-42`). The cross-cutting
issues from `report.md` (missing SSIM, no shared baseline, single-permutation
shuffle) were largely addressed. This is good experimental hygiene.

---

## 2. RESOLVED CRITICAL — Exp 1 (shuffle observables) was internally contradictory

This was the most important finding of the review and has now been resolved.

Two files in the **same experiment folder** report incompatible numbers:

| Source | Normal PSNR | Shuffled PSNR | Drop |
|---|---:|---:|---:|
| `shuffle_eval_summary.json` (the actual eval output) | 32.24 dB | **32.04 dB** | **−0.19 dB** |
| `INTERPRETATION.md` / `report.md` table | 32.20 dB | **19.70 dB** | **−12.5 dB** |

The summary JSON also records `shuffled_mse_vs_normal = 2.3e-5` — i.e. the
"shuffled" reconstruction is **essentially identical** to the unshuffled one.
A −0.19 dB drop and a 2.3e-5 MSE delta mean the shuffle **did almost nothing**,
which is the *opposite* of the −12.5 dB "load-bearing observables" conclusion
written up in `report.md` and `INTERPRETATION.md`.

The shuffle path was checked end-to-end: the decoder receives
`observables[perm]` while positions remain fixed. A verifier then loaded the
saved decoder and observable tensor and evaluated five non-identity permutations.
The verified mean PSNR drop is **0.301 ± 0.054 dB**, with range **0.236 to
0.366 dB**. Therefore the JSON was the correct behavior and the −12.5 dB table
was stale.

**Required interpretation:** Exp 1 is weak or negative evidence for
observable-position load-bearing behavior. It must not be cited as a large
shuffle degradation.

*(Note: Exp 2's zero-latent JSON is self-consistent — 32.24 → 14.85 dB, −17.4 dB —
so the position-memorization ruling-out still stands. Exp 1 is the broken one.)*

---

## 3. The core scientific problem — the quantum component is not pulling its weight

Reading the shared-seed-42 runs at face value, the ablations **converge on a
negative result for the strong claim**:

| Experiment | PSNR | SSIM | vs baseline (32.24 dB) | Implication |
|---|---:|---:|---|---|
| **Baseline (full HVK)** | 32.24 | 0.9919 | — | reference |
| Freeze **quantum** (VQC+J fixed random) | **33.11** | 0.9934 | **+0.87 dB better** | trained quantum params are *not needed* |
| Classical `Linear(6→27)+tanh` replacement | **33.45** | 0.9939 | **+1.21 dB better** | VQC beaten by a trivial classical map |
| Classical-matched | 33.19 | 0.9935 | +0.95 dB | same story |
| No entanglement (rotations only) | 32.62 | 0.9926 | +0.38 dB | entanglement adds nothing |
| No MPS (patch stats) | 32.69 | 0.9927 | +0.45 dB | MPS encoder adds nothing |
| No energy loss | 33.30 | 0.9937 | +1.06 dB | Hamiltonian term is inert/mildly harmful |
| Random VQC (untrained) | 25.98 | 0.9656 | −6.26 dB | a *structured* map is needed, not a *quantum* one |
| Freeze classical (decoder frozen) | 10.93 | 0.021 | −21 dB | all learning is classical |

**Every targeted quantum ablation either matches or *beats* the full model.**
The only components that are load-bearing are the **classical decoder/projections**
(freeze-classical collapses to noise) and having *some* structured latent
(random-VQC is bad). Nothing here isolates a benefit from: trained VQC weights,
entanglement, the MPS tensor network, or the Heisenberg energy term.

This is a clean, defensible **negative result** — but it directly contradicts the
paper's intended narrative. It cannot be written up as a quantum-advantage paper on
this evidence. The student's own `INTERPRETATION.md` honestly says as much; the
supervisor-level framing must follow that honesty.

---

## 4. Generalization — the single most damaging result

| Run | PSNR | SSIM |
|---|---:|---:|
| Second image (inference, no retrain) | **7.78** | **0.0196** |
| Multi-image (trained across images) | 28.31 | 0.9862 |

Zero-shot transfer to a second image is **7.78 dB / SSIM 0.02** — that is *below*
the random-latent noise floor (~12 dB). The trained model has **memorized one
image**. This means every other experiment is measuring a **per-image optimization
procedure**, not a learned representation. Per `report.md`'s own criterion, this
"paper framing must change." The multi-image run (28.31 dB) is the only path to a
real contribution and should become the primary setup, not an afterthought.

---

## 5. Methodological issues that block publication-quality claims

1. **n = 1 everywhere.** Single seed, single image, single permutation. All the
   quantum-vs-classical gaps (0.4–1.2 dB) are *within plausible seed noise* and
   several go the "wrong" way. No error bars ⇒ **none of these differences are
   statistically established.** Need ≥5 seeds with mean ± std before any comparison
   is reportable.

2. **PSNR differences are tiny and non-monotonic.** χ1 (32.95) > χ4 (32.24);
   q8 (33.07) > q6 (32.24); no-mps ≈ mps. A monotonic trend is absent, which is
   itself evidence the quantum/TN knobs aren't the driver — consistent with noise.

3. **Energy/order-parameter "phase transition" reporting is unconvincing.** The
   detector fires in *every* run (including classical-replacement with energy
   identically 0) using a median+2σ threshold on a tiny susceptibility signal
   (~0.006 vs 0.002). A "phase transition" that appears even when the Hamiltonian
   is switched off is a threshold artifact, not physics. Do not present this as
   evidence of Hamiltonian dynamics.

4. **Hardcoded absolute paths leak** (`/home/adminpc/Desktop/HVK/...` inside
   `hvk1d_standard_vs_symmetric_metrics.json`). Cosmetic, but shows outputs aren't
   fully portable/reproducible.

5. **Baseline drift already noted** is now mostly fixed via shared-seed-42 — good.

---

## 6. What the results DO support (the honest, defensible claims)

- The classical decoder + projections are necessary and sufficient for
  single-image reconstruction (freeze-classical fails; freeze-quantum succeeds).
- A *structured* latent is required (random-VQC is poor), but it need not be
  quantum (classical tanh is as good or better).
- Position alone is insufficient (Exp 2 zero-latent, self-consistent).
- Convergence: ~240 steps reaches the 120-step-plateau region; 120 was mildly
  underfit (28.75 → 32.55 dB from 120→240). **Re-run core comparisons at ≥240
  steps** — the 120-step numbers above understate several variants.

---

## 7. Required actions before this is publishable

**Blocking:**
1. Resolve the Exp 1 contradiction; verify the shuffle actually perturbs the
   decoder input. Re-run and report true drop with fixed-point count over ≥5
   permutations.
2. Add multi-seed runs (≥5) with mean ± std for every comparison in §3. Report
   whether any quantum component gap is significant; on current evidence it is not.
3. Re-run core ablations at 240 steps (matched convergence) so comparisons are fair.

**Framing:**
4. Accept the negative result: on this benchmark the VQC, entanglement, MPS, and
   energy term are not demonstrably contributing. Either (a) reframe as an honest
   study of *what does and doesn't help* in a hybrid pipeline, or (b) pivot to the
   multi-image regime where a genuine learned representation might exist, and
   re-run the quantum ablations there — that is the only setting where a quantum
   advantage could still show up.
5. Drop or heavily caveat the "phase transition" narrative until it survives a
   control (it currently fires with the Hamiltonian off).

---

## 8. Bottom line for the student

Excellent execution and completeness — the suite is thorough and the local
`INTERPRETATION.md` notes are refreshingly honest about the negative outcomes. But
two things must be fixed before anything is written into a paper: (1) the Exp 1
shuffle number is self-contradictory and one version is wrong, and (2) with single
seed / single image, none of the quantum-vs-classical differences are established,
and those that exist point *against* a quantum advantage. The current evidence
supports a careful negative/ablation paper or a pivot to the multi-image setting —
not a quantum-advantage claim.

---

## 9. Review of the new `main2/newHVK/` second-generation study

The student rebuilt the study to address §5's statistical objections. Credit
where due — this is a real step up in rigor. But the results split cleanly into
**one honest negative result (trustworthy)** and **one circular "advantage"
(not trustworthy)**. Do not conflate them.

### 9a. What genuinely improved (keep this)

- **Multi-seed with error bars** (`full_ablation_summary.csv`, 5 seeds, mean±std).
- **Real held-out CIFAR-10**, 20 images, train/test split (`real_cifar_holdout`).
- **Same-width classical controls** and **resource matching** (`resource_comparison.json`:
  all models at feature_dim 32, 2112 readout params — a fair comparison at last).
- **Finite-shot noise sweep** (128→8192 shots) and an **IBM circuit probe**.
- Honest claim boundaries written into every README ("candidate", "not a
  hardware quantum-advantage proof").

### 9b. The honest result — held-out CIFAR (TRUSTWORTHY, and it is negative)

From `real_cifar_holdout_summary.csv` (20 held-out images, mean PSNR):

| Model | PSNR | SSIM |
|---|---:|---:|
| local-observables-only / raw-linear-classical | **20.66** | 0.862 |
| zz-only | 20.52 | 0.858 |
| no-entanglement | 20.24 | 0.851 |
| shuffled-pair-observables | 20.17 | 0.848 |
| **newHVK-real-cifar (entangling)** | **20.07** | 0.846 |
| strict-classical-rff | 17.06 | 0.706 |
| random-vqc | 10.99 | 0.016 |

On **real images**, the entangling quantum model (20.07 dB) is **statistically
tied with — and nominally slightly below — a plain local/linear classical model
(20.66 dB)**. Entanglement, zz, and pair observables buy nothing. The student's
own Q1 addendum says exactly this: *"the real held-out CIFAR test does not
establish quantum advantage."* **This is the correct, publishable conclusion**
and it confirms report_2 §3 on independent (real, held-out, multi-seed) data.

### 9c. The circular result — CIFAR "nonlocal advantage" (NOT TRUSTWORTHY)

`cifar_nonlocal_pair_summary.csv` reports the entangling model at
**R² = 0.99999999999997, PSNR = 120 dB, MSE ≈ 4e-15** — a machine-precision
*perfect* fit, while every control sits at R² ≈ 0.2 or worse. That gap is not a
representational advantage; it is **label leakage baked into the task design.**

Concretely, in `run_newhvk_suite.py`:
- The **target** (`target_from_standardized`, lines 1374–1384) is a fixed list
  of six feature products, e.g. `left[:,0]*right[:,0]`, `left[:,1]*right[:,1]`,
  `left[:,4]*right[:,5]`, …
- The **entangling features** (`cifar_pair_entangling_features`, lines 1400–1407)
  contain those **exact same six products** as explicit columns.

So the target is a *linear function of features handed only to the entangling
model*, and a linear ridge regressor inverts it to numerical zero error. The
controls (`raw-linear`, `no-entanglement`, `left/right-only`) are denied those
precise product columns, so they **cannot** fit it — by construction, not by
physics. Any model given the product columns (including a classical one) would
score R²=1.0; the "entanglement" framing is incidental.

**This is a rigged benchmark and must not appear in a paper as evidence of
quantum/entanglement advantage.** A reviewer who reads the dataset code will
reject the paper on this point. At minimum: the target must depend on the raw
patch pixels through a process the model does *not* receive as a pre-computed
feature, and *every* model must have equal opportunity to learn the nonlocal
product (e.g. give the classical control the same product basis, or make the
target a genuine held-out image quantity).

### 9d. Verdict on newHVK

The workspace is methodologically much stronger and its **real-data conclusion
is sound and negative**: on held-out CIFAR, the quantum/entangling components do
not beat matched classical controls. The one positive ("nonlocal advantage") is
an artifact of target–feature leakage and carries no evidence. Net position is
unchanged from §7: this supports an honest ablation / negative-result paper, not
a quantum-advantage claim. Delete or redesign the nonlocal diagnostic before it
does reputational damage.
