# TODO — Blocking tasks before the paper goes out

Two things must be fixed. Both are **blocking**: no result table can be trusted
until these are done. Do them in order.

---

## Task 1 — Resolve the Exp-1 shuffle contradiction

**Problem.** The same experiment reports two incompatible numbers:

| Source | Shuffled PSNR | Drop |
|---|---:|---:|
| `shuffle_eval_summary.json` (actual eval output) | 32.04 dB | **−0.19 dB** |
| `INTERPRETATION.md` / `report.md` table | 19.70 dB | **−12.5 dB** |

The JSON also says `shuffled_mse_vs_normal = 2.3e-5` → the shuffle changed
almost nothing. One of these is wrong. Find out which.

**Do this:**

1. Open the shuffle code path. Confirm the permuted observable tensor is the
   **exact object fed to the decoder** — not a copy that gets discarded.
2. Assert the permutation is **non-identity**: log how many indices actually
   moved (fixed-point count). A no-op shuffle would explain the −0.19 dB.
3. Re-run the shuffle eval over **≥5 different random permutations**, not one.
   Report mean ± std of the PSNR drop.
4. Write the **true** number into `report.md`, `report_2.md`, and
   `INTERPRETATION.md`. Delete the stale one.

**Deliverable:** one short note in the Exp-1 folder stating the verified drop,
the fixed-point count, and which of the two old numbers was correct.

> ⚠️ If the true drop is ~0.2 dB, then observables are **NOT** load-bearing at
> eval — that flips the Exp-1 conclusion. That is fine; just report it honestly.

---

## Task 2 — Multi-seed at matched convergence for the core comparisons

**Problem.** The core ablation table is **single-seed** and trained for only
**120 steps**, which is underfit (120→240 steps: 28.75 → 32.55 dB). Sub-dB
quantum-vs-classical gaps at 120 steps / one seed prove nothing.

**Do this:**

1. Re-run **every** core comparison at **≥240 steps** (matched budget for all
   variants — same step count, same LR, same schedule).
   Core set: baseline, freeze-quantum, freeze-classical, no-entanglement,
   no-MPS, no-energy, classical-replacement, classical-matched, random-VQC.
2. Run each with **≥5 seeds**. Report **mean ± std** PSNR and SSIM.
3. In the table, mark any quantum-vs-classical gap **smaller than 1 std** as
   "not significant" — do not claim it.

**Deliverable:** one updated table (mean ± std, 5 seeds, 240 steps) that
replaces the old single-seed table everywhere it appears.

> On current evidence, expect the quantum gaps to stay inside the noise band.
> That is the honest negative result the paper is built on — report it plainly.

---

## Task 3 — Flag the "quantum advantage" result (do NOT claim it)

> This is the `cifar_nonlocal_advantage` result (commit `c4c2e2c`) — the
> R²≈1.0 / PSNR=120 dB number. **It is not a quantum advantage. Do not report
> it as one.** Read this before building any claim on it.
>
> **We are NOT claiming quantum advantage in the paper.** Don't delete the
> result yet — just leave it flagged as circular/invalid so nobody mistakes it
> for a real finding. It stays out of the results and out of any claim.

**Why it's circular.** You changed the *task*, not the architecture. The target
(`run_newhvk_suite.py:1374-1385`) is built from the exact pair-product features
handed **only** to the entangling model (`run_newhvk_suite.py:1396-1413`). So the
target is a linear function of columns the classical controls never get. A linear
fit then inverts it to zero error → R²=1.0, MSE≈4e-15. **Any** model given those
columns (classical included) would score 1.0. The tell: R²=0.9999999999999 /
120 dB is machine precision — real advantages never look like that; that's a
label leak.

**The real result is already in your suite.** On the honest held-out CIFAR test,
the entangling model gets **20.07 dB vs 20.66 dB** for a plain local-linear
classical — **tied/slightly behind. No advantage.** That is the trustworthy one
(→ Task 4).

**Do this (pick one):**
- **Delete it** from all results and claims, **or**
- **Redesign it** so (a) the target depends on the raw patch pixels through a
  process the model does NOT receive as a precomputed feature, and (b) **every**
  model (including classical controls) gets equal access to the nonlocal basis.

**Do this (pick one):**
- **Delete it** from all results and claims, **or**
- **Redesign it** so (a) the target depends on the raw patch pixels through a
  process the model does NOT receive as a precomputed feature, and (b) **every**
  model (including classical controls) gets equal access to the nonlocal basis.

**Deliverable:** either the diagnostic is removed everywhere, or a corrected
version where the classical control can compete on equal footing.

---

## Task 4 — Make held-out CIFAR the primary result

The single-image runs measure per-image memorization (zero-shot = 7.8 dB).
The **held-out CIFAR-10** multi-seed result is the only trustworthy setting.

**Do this:**
1. Promote held-out CIFAR (20 images, multi-seed) to the main results table.
2. Confirm resource matching: all models at equal feature width (32) and equal
   readout params (2112). Log the config next to the table.
3. State the conclusion plainly: entangling model ties/trails local-linear
   classical — no advantage.

---

## Task 5 — Drop the "phase transition" narrative (until it survives a control)

The energy/order-parameter detector fires in **every** run — including the
classical-replacement run where energy is identically 0. A median+2σ threshold
on a tiny signal (~0.006 vs 0.002) is a threshold artifact, not physics.

**Do this:** either remove the phase-transition claim, or show it fires ONLY
when the Hamiltonian is on and NOT when it is off. If it fails that control,
delete it.

---

## Task 6 — Reproducibility cleanup

1. Remove hardcoded absolute paths (e.g. `/home/adminpc/Desktop/HVK/...` in
   `hvk1d_standard_vs_symmetric_metrics.json`). Use relative paths.
2. Pin seeds and log them in every results file.
3. One script per table: make each paper table regenerable from a named script.
4. Write a short `REPRODUCE.md`: which script produces which table/figure.

---

### Done checklist
- [ ] **T1** Exp-1 shuffle path verified end-to-end (tensor reaches decoder)
- [ ] **T1** Exp-1 re-run over ≥5 permutations; true drop written to all 3 files
- [ ] **T2** Core ablations re-run at ≥240 steps, ≥5 seeds
- [ ] **T2** Table updated to mean ± std; within-noise gaps marked not significant
- [ ] **T3** Nonlocal benchmark deleted or redesigned (no leakage)
- [ ] **T4** Held-out CIFAR promoted to primary result; resource matching logged
- [ ] **T5** Phase-transition claim passes an on/off control, or removed
- [ ] **T6** Absolute paths removed; seeds pinned; `REPRODUCE.md` written

---

**Priority order:** T1 and T2 are blocking (nothing is trustworthy without them).
T3 is next (a reviewer reading the code will reject on it). T4–T6 finish the
project into a clean, publishable negative-result study.
