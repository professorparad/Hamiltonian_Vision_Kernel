# HVK submission bundle — IEEE Transactions on Quantum Engineering

**Manuscript:** *Hamiltonian Vision Kernel: Symmetry-Pooled Quantum Correlator
Features with Hardware Validation*
**Authors:** Sparsho Chakraborty (IIT Bhubaneswar), Siddhartha Patra (CQuERE, TCG CREST)
**Route:** Traditional (non-open-access). arXiv preprint intended on acceptance.
**Assembled:** 2026-08-13

## Contents

| Path | What it is |
|---|---|
| `pdf/paper_hvk.pdf` | Main manuscript (11 pp., IEEEtran) |
| `pdf/supplementary_study.pdf` | Supplementary material (26 pp.) |
| `pdf/cover_letter.pdf` | Cover letter to the Editor-in-Chief |
| `pdf/author_contributions.pdf` | CRediT contribution statement |
| `source/` | LaTeX sources + `figures/` — compiles standalone with two `pdflatex` passes, no external files needed |
| `REPRODUCE.md` | One row per table/figure cluster: driver script → output artifact |
| `submission_claim_audit.json` | Machine-readable provenance for every numerical headline claim, plus `withdrawn_claims` |

## What this paper claims — and what it does not

HVK is presented as a **new hybrid quantum–classical image-reconstruction
architecture that works**, is **competitive** with resource-matched classical
baselines, runs on **real hardware**, and adds **capabilities** classical maps do
not natively expose. **No quantum advantage is claimed anywhere.**

Three scoping rules are load-bearing and were applied consistently:

1. **"Competitive" means TOST equivalence at a pre-declared ±1 dB margin — never
   "beats."** Six of eight resource-matched controls are statistically equivalent
   to HVK2D. The two that are not are controls HVK2D is *ahead* of.
2. **The held-out CIFAR-10 result is a loss for HVK and is reported as one**
   (classical 18.80 dB vs. HVK2D 18.12 dB). It appears in the abstract, results,
   and conclusion, not only in the supplement.
3. **$D_4$ equivariance is a design-correctness property, not a quantum result.**
   The same Reynolds averaging applied to a purely classical feature map is
   equally exact ($8.94\times10^{-17}$ vs. $9.47\times10^{-17}$).

## Withdrawn before submission

All change-point / "critical epoch" / "critical temperature" detection material was
**removed**, not merely rescoped. No sharp transition exists in any trace — only
smooth drift with run-to-run fluctuation — and the apparent detection was an
artifact of a within-run median+2σ threshold computed from the very trace it
scored. The negative control settles it: applied to a classical-replacement variant
whose Hamiltonian energy is identically zero throughout training, the rule fires in
**4/4** cases, exactly as often as the Hamiltonian-regularized model's **4/4**.

What survives is the honest remainder: the on/off negative-control table (the
disproof itself), the descriptive tally shown only beside it, and one $R_{ES}$
trace recaptioned as a plain interpretability readout. Twenty-seven orphaned
figures were deleted. `tests/test_submission_claim_audit.py::test_no_transition_claims_in_manuscripts`
fails the build if transition language reappears without a disclaimer.

## Verification status at assembly

- Both manuscripts: **0 errors, 0 undefined references, 0 undefined citations** over two `pdflatex` passes.
- `source/` recompiles standalone from a clean directory.
- `ruff check .` — all checks passed.
- `pytest tests/` — **19 passed**.

## Still owned by the supervisor (not filled here)

- ORCID iD for Siddhartha Patra — `cover_letter.tex` line ~68 (`[FILL]`).
- 2–3 suggested reviewers — `cover_letter.tex` (`[FILL]`).
- CRediT split confirmation — `author_contributions.tex` (`[FILL]` ×2).
- Zenodo DOI minted from a tagged release, added to `CITATION.cff`.
- Confirmation of the TQE article type and length limit.
