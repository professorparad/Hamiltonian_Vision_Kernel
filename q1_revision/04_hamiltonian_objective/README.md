# Hamiltonian objective

## Reviewer ask

"Removing the energy loss improves reconstruction. This weakens both 'Hamiltonian' and
'physics-informed' as central performance motivations." Either redesign the objective so
it contributes measurably, or remove "physics-informed" performance implications and
present energy strictly as an interpretability diagnostic.

## What already exists

The paper's own ablation (Table X, "Hamiltonian and observable-sector controls") already
shows this honestly: removing the energy loss improves PSNR from 32.24 to 33.30 dB. This
is not a new finding to establish — the reviewer's ask is about what to *do* about it,
not about discovering it.

## Two paths (need a decision, not more code, to start)

1. **Reframe only** (cheap, text-only): stop describing the Hamiltonian term as a
   performance-improving inductive bias anywhere in the paper; present it strictly as an
   interpretability/diagnostic signal (which is what the existing R_ES / energy-tracking
   diagnostics already effectively use it for). Fits naturally alongside
   `../01_narrative_reframe/` and `../08_title_and_claims/`.
2. **Redesign the objective** (real experiment): try alternative energy formulations
   (e.g. an unsigned/magnitude energy term, a contrastive energy objective — note Table X
   already tests a "Contrastive Hamiltonian core" variant at 32.84 dB, still below the
   no-energy-loss baseline) to see if any variant actually helps. Given the existing
   ablation already tried several variants without success, this path has a real risk of
   spending compute to reconfirm the same negative result.

## Status (updated 2026-07-28)

Path 2 attempted, confirmed with project owner. Root-cause diagnosis: `Jx/Jy/Jz`
couplings (HVK1D) / `j_2d` (HVK2D) are free parameters that feed only the energy loss,
never the decoder — under a linear energy loss they can grow unboundedly to cheaply
minimize the loss with no reconstruction benefit, fighting the reconstruction gradient
on shared weights. Fix: bound the couplings (`tanh`) and use the already-implemented but
previously-untested `positive` (squared) energy-loss mode. Built in new folders
`Main_new/` (HVK1D) and `Main_new2/` (HVK2D) — see those folders' READMEs for full
detail and code.

**Important caveat found during validation**: re-running Table X's *unmodified*
baselines (legacy/no-energy-loss/contrastive) at the documented protocol reproduces
PSNR values 6-12 dB higher than published, across the board — not just for the new
variant. The original generating script appears to have been lost (gitignored,
referenced in commit messages but never actually committed). Decision (confirmed with
project owner): report a fresh, self-consistent same-environment comparison instead of
Table X's numbers, with this gap disclosed explicitly rather than silently overwritten.
See `Main_new/README.md` for the full writeup of this finding.

**v1 result** (bounded couplings + `positive` loss, energy still a side-channel loss
only): ties "no energy loss" (44.56 vs 44.56-44.69 dB at seed 42) — stops the Hamiltonian
from *hurting*, doesn't make it *help*. Root cause: energy never reaches the decoder, so
it structurally cannot improve reconstruction, only compete with it or sit neutral.

**v2 fix**: feed energy into the decoder as an actual input (`use_energy_feature=True`,
see `Main_new/README.md`). Full 3-seed result vs "no energy loss":

| Seed | No energy loss | Energy fed to decoder | Delta |
|---|---|---|---|
| 42 | 44.56 dB | 45.81 dB | +1.25 dB |
| 43 | 43.77 dB | 44.28 dB | +0.51 dB |
| 44 | 45.33 dB | 40.25 dB | -5.08 dB |

**Not a robust win.** 2/3 seeds improve, 1/3 regresses badly enough that the mean delta
across seeds is negative (~-1.1 dB). Feeding energy into the decoder increases outcome
variance more than it reliably improves the mean. This is a genuinely different, weaker
finding than "the Hamiltonian now helps" — it's "the Hamiltonian *can* help, seed-
dependently, and can also hurt more than the old bug did." Chasing an apparent bug in the
seed-44 result (it initially looked suspiciously like a stale-cache artifact) cost
significant time before an isolated rerun with explicit `PYTHONPATH` confirmed it was a
genuine, reproducible result, not a bug — a useful lesson in not assuming "looks like a
bug" without an unambiguous isolated retest.

**Still open**: this needs a larger seed count (5+) before either claim ("the Hamiltonian
helps" or "it's neutral/diagnostic-only") can be made with confidence. Current best
framing given the data in hand: energy-as-decoder-feature is a promising *direction*,
not yet a validated result — present it as exploratory in the paper, with the honest
per-seed variance shown, rather than as a headline fix. Path 1's conservative framing
(energy as diagnostic, not performance claim) remains the safer default until more seeds
land.
