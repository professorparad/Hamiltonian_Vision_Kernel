# Quantum Contribution Ablation — Critical Review (report_2)

**Reviewer role:** publication-quality scrutiny of the results in
`experiments/quantum_contribution/results/`.
**Date:** 2026-07-10
**Verdict in one line:** The student ran an impressively complete ablation suite
(all 10 planned experiments plus extras), and it is thorough and well-organized —
**but the results, read honestly, do not support a quantum-advantage claim, and
one headline number (Exp 1) is internally contradictory and cannot be trusted as
reported.**

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

## 2. CRITICAL — Exp 1 (shuffle observables) is internally contradictory

This is the most important finding of the review and must be resolved before any
of Exp 1's conclusion is used.

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

**Only one of these can be true.** Likely explanations to check:
- The shuffle permutation was applied to a copy that was not actually fed to the
  decoder (no-op shuffle), so the JSON (−0.19 dB) is the real behavior and the
  −12.5 dB table is stale/hand-entered from an earlier run.
- Or the JSON is from a broken run and the −12.5 dB is correct.

**Action required:** Do not cite Exp 1 until the shuffle path is verified end-to-end
(assert the permutation is non-identity, log fixed points, confirm the permuted
tensor reaches the decoder). If the true drop is ~0.2 dB, then **Exp 1 actually
shows observables are NOT load-bearing at eval** — a much weaker and very different
result. This flips the paper's opening sanity check.

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
