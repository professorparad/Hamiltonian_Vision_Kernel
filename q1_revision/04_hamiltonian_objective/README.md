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

**Honest current result** (single seed, being extended to 3): the `positive` energy-loss
fix stops the Hamiltonian term from *hurting* (it no longer trails "no energy loss" by
~6 dB the way the legacy linear mode does), but at one seed it only ties with removing
the energy loss entirely rather than clearly beating it. Bounding the couplings on top
of `positive` mode doesn't add anything further at that seed. Multi-seed check (43, 44)
in progress to see if this holds or if noise is doing the work.

**Still open**: which framing (path 1 vs 2) the paper actually uses depends on how the
multi-seed result lands. If `positive`+bounded ties "no energy loss" robustly across
seeds, the honest framing is still closer to path 1 (energy is a diagnostic, not a
performance win) but with a corrected, no-longer-actively-harmful default. If it
robustly wins, path 2's stronger claim becomes defensible. Either way, this is a more
defensible result than Table X's original story.
