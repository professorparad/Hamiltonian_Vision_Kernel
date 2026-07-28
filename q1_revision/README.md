# Q1 Revision Tracker

Source: external reviewer assessment, 2026-07-27. Overall score 5.5/10 for Q1-journal
readiness ("major revision or rejection at a strong Q1 venue in present form").
Full review text: see `reviewer_assessment.md` in this folder.

This directory tracks the reviewer's "Highest-impact revisions" list, one subfolder per
item. Each subfolder has its own `README.md` with: what the reviewer asked for, what
already exists in the repo (so we don't rebuild what's there), what's missing, and the
concrete next step. Code changes happen in their normal repo locations (`Main2/newHVK`,
`Baselines/cifar10_comparisons`, `IBM_Cloud`, `latex_outputs`) — these folders are an
index and plan, not a new code location.

## Status board

| # | Folder | Revision item | Status |
|---|--------|----------------|--------|
| 1 | [01_narrative_reframe](01_narrative_reframe/) | Reframe central claim + title | Not started |
| 2 | [02_dataset_level_generalization](02_dataset_level_generalization/) | Real train/val/test generalization study | **In progress** (background sweep running) |
| 3 | [03_d4_end_to_end_integration](03_d4_end_to_end_integration/) | Integrate D4 pooling into trainable pipeline | Not started |
| 4 | [04_hamiltonian_objective](04_hamiltonian_objective/) | Fix or reframe the Hamiltonian energy term | Not started |
| 5 | [05_hardware_study_strengthening](05_hardware_study_strengthening/) | Repeated jobs, mitigation, uncertainty intervals | Not started — needs IBM Quantum quota decision |
| 6 | [06_classical_baselines](06_classical_baselines/) | Stronger classical comparison models | Partially exists — gap analysis done |
| 7 | [07_phase_transition_scope](07_phase_transition_scope/) | Reduce/reframe phase-transition section | **In progress** (qubit-energy sweep + combined-figure work) |
| 8 | [08_title_and_claims](08_title_and_claims/) | Shorten title, audit claim language | Not started |

## Priority notes

- Items 2 and 7 are already underway (see their folders for live status and pointers to
  the running background jobs and generated figures).
- Item 5 (hardware strengthening) needs a decision before any code is written: how much
  IBM Quantum free-tier quota is left, and whether more hardware time is available at
  all. Everything else in this tracker can proceed on local compute.
- Items 1 and 8 are text-only (no new experiments) and can be done any time once the
  experimental items above land, since the narrative should reflect final results, not
  the other way around.
- Item 6 mostly reuses `Baselines/cifar10_comparisons/` scripts already in the repo; the
  gap is smaller than it first appears.
