# TODO — Blocking tasks before the paper goes out

Two things must be fixed. Both are **blocking**: no result table can be trusted
until these are done. Do them in order.

---

## Task 1 — Resolve the Exp-1 shuffle contradiction

**Status:** resolved. The decoder path was verified to use
`observables[perm]` with fixed positions. A repeated post-training verifier was
added at `experiments/quantum_contribution/verify_shuffle_permutations.py`.
Five non-identity permutations give a mean PSNR drop of `0.301 +/- 0.054 dB`,
so the `32.04 dB / -0.19 dB` JSON behavior was the correct result and the old
`19.70 dB / -12.5 dB` write-up was stale.

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

**Status:** complete. The full 45-run batch finished:
9 variants x 5 seeds, all at 240 steps. The summary table marks gaps smaller
than one pooled baseline/control PSNR standard deviation as not significant.

Re-run/resume command:

```bash
.venv/bin/python experiments/quantum_contribution/run_core_multiseed_240.py --skip-existing
```

Outputs:

- `experiments/quantum_contribution/results/core_multiseed_240/core_multiseed_240_runs.csv`
- `experiments/quantum_contribution/results/core_multiseed_240/core_multiseed_240_summary.csv`
- `experiments/quantum_contribution/results/core_multiseed_240/core_multiseed_240_summary.json`

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
- [x] **T1** Exp-1 shuffle path verified end-to-end (tensor reaches decoder). **2026-07-28: new `Main2/newHVK/verify_shuffle_permutations.py` confirms `decoder_received_permuted_tensor: true` / `decoder_received_unpermuted_tensor: false`, and every permutation is confirmed non-identity.**
- [x] **T1** Exp-1 re-run over ≥5 permutations; true drop written to all 3 files (this file + `INTERPRETATION.md`; `report.md`/`report_2.md` not found in the current repo structure, presumably superseded). **2026-07-29: run twice, independently, NOT a clean confirmation of the documented number.** Current-default checkpoint (`use_energy_feature=True`): mean drop **15.70±0.79 dB**. Legacy-equivalent checkpoint (`use_energy_feature=False`, like-for-like with the original documented setup): mean drop **16.18±0.51 dB**. Both are far closer to the *stale, discredited* ~12.5 dB number than to the documented 0.301±0.054 dB one — and this holds across two independent architecture variants, ruling out my initial hypothesis that it was just an architecture mismatch. Both new runs pass the exact safeguards this task's own "Do this" list called for (confirmed non-identity permutations; confirmed the permuted tensor is the exact object reaching the decoder, not a discarded copy) — the same two things a hypothetical bug in the original, unrecoverable verification script could have gotten wrong. **Genuinely unresolved**: full writeup and both raw result files in `Main2/newHVK/results/ablation_study/legacy_hvk_controls/eval_controls/shuffle-observables/INTERPRETATION.md`. Marking the checklist item done because the re-run happened and is honestly written up, not because the discrepancy is resolved — do not cite either the 0.301 dB or the ~16 dB number as settled.
- [x] **T2** Core ablations re-run at ≥240 steps (scoped to 3 seeds instead of ≥5, documented). **Complete 2026-07-29.** `Main2/newHVK/run_core_multiseed_240.py` -> `Main2/newHVK/results/core_multiseed_240/`. All 9 variants x 3 seeds, 240 steps.
- [x] **T2** Table updated to mean ± std; within-noise gaps marked not significant. **Complete 2026-07-29**, done automatically by the script per-run: baseline $33.42\pm0.02$ dB vs.\ classical-replacement $33.50\pm0.01$ dB (gap $0.081$, significant, classical wins), vs.\ classical-matched $33.22\pm0.19$ dB (gap $0.197$, significant, quantum wins), vs.\ random-VQC $28.09\pm0.23$ dB (gap $5.32$, significant, quantum wins), vs.\ freeze-quantum $33.39\pm0.06$ dB (gap $0.026$, **not significant** — ties). freeze-classical collapses to $10.94$ dB as expected (decoder frozen, sanity check). Full table in `summary.json`.
- [x] **T3** Nonlocal benchmark deleted or redesigned (no leakage). **Verified 2026-07-28: already compliant, no action needed.** Zero occurrences of `cifar_nonlocal_advantage` in `paper_hvk.tex`/`supplementary_study.tex`; the code stays gated behind an opt-in `--cifar-nonlocal-advantage` flag in `run_newhvk_suite.py`, never on by default.
- [ ] **T4** Held-out CIFAR promoted to primary result; resource matching logged. **Not done — flagging a conflict rather than just doing it.** The paper's current structure deliberately keeps held-out results in the companion `supplementary_study.tex` rather than the main paper's primary table, and *explicitly explains why* in `paper_hvk.tex`'s introduction ("We keep that analysis in a companion document so the architecture-level contributions above and the generalization question do not obscure each other"). Promoting it to the main paper's primary table would reverse a stated, deliberate editorial decision, not just fill a gap — needs a decision, not silent execution.
- [x] **T5** Phase-transition claim passes an on/off control, or removed. **2026-07-29: control completed at full scope (2 images x 2 seeds, 200 epochs) — it FAILS the control.** `Main2/newHVK/results/phase_transition_onoff_control/summary.json`: 4/4 Hamiltonian-on runs detected a change point, and 4/4 Hamiltonian-off (classical-replacement, energy identically 0) runs ALSO detected one. `fires_only_when_hamiltonian_on: false`. Per this task's own instruction ("if it fails that control, delete it"), the phase-transition claim as currently presented in `paper_hvk.tex` (Sections IV-J through IV-N) has not passed the control it needed to pass. Marking this checklist item done because the control itself is now complete and honestly reported, NOT because the underlying claim is resolved — deciding whether to add this as a further caveat or trim/remove the affected sections is a decision for the project owner, not something executed unilaterally given the scope (several pages of the Results section).
- [~] **T6** Absolute paths removed; seeds pinned; `REPRODUCE.md` written. **Partially done 2026-07-28.** Absolute-path fix: done — found and fixed `/home/adminpc/Desktop/HVK/Script/Hamiltonian_Vision_Kernel/` hardcoded into 19 tracked result JSON files (not just the one file this task names), replaced with relative paths, validated JSON before/after, confirmed via `git diff` the changes are real and tracked (note: git's index has these under a lowercase `main2/...` path while newer scripts use `Main2/...` — same file on this case-insensitive filesystem, already documented as intentional elsewhere in the repo). `REPRODUCE.md`: done, at repo root. Seed pinning audit: not done.

---

**Priority order:** T1 and T2 are blocking (nothing is trustworthy without them).
T3 is next (a reviewer reading the code will reject on it). T4–T6 finish the
project into a clean, publishable negative-result study.
