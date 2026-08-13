# Reproducing the HVK paper's results

One row per table/figure cluster: driver script → output directory. Grouped by
paper section rather than by individual table number, since most sections'
tables/figures share one driver script and output directory. See
`project_artifacts/submission_claim_audit.json` for numeric cross-checks of
the headline claims, and `q1_revision/` for the provenance write-ups behind
the two newest additions (dataset-level generalization, Hamiltonian
objective).

## `paper_hvk.tex`

| Section | Result | Script | Output |
|---|---|---|---|
| §II, Fig. 1 | HVK1D/HVK2D ansatz circuits | `IBM_Cloud/generate_ansatz_figures.py` (rebuilds the ansatz gate-for-gate in Qiskit from the real training circuit, `Main_new/src/quantum/circuit.py::VQC`; see script docstring for the verification method) | `latex_outputs/paper_latex/figures/hvk1d_ansatz.pdf`, `hvk2d_ansatz.pdf` |
| §IV-A, Table II | Same-set reconstruction, 6 datasets | `Main2/newHVK/run_full_dataset_sameset.py --datasets cifar10 mnist fashion-mnist pathmnist --images-per-dataset 3` then `--datasets bloodmnist --images-per-dataset 2` (script overwrites its output rather than merging, so run in batches and merge by hand; PneumoniaMNIST already present at n=2) | `Main2/newHVK/results/full_dataset_sameset/summary.json` |
| §IV-B | Zero-shot / multi-image adaptation (8.31 / 28.63 dB) | `Main2/newHVK/run_zero_shot_generalization.py --epochs 100` | `Main2/newHVK/results/zero_shot_generalization/summary.json` |
| §IV-C | Real IBM hardware reconstruction pilot | `IBM_Cloud/run_hvk_hardware_reconstruction.py` (Monalisa), `IBM_Cloud/run_hvk2d_cifar_hardware_reconstruction.py` (CIFAR) | `IBM_Cloud/outputs/hardware_reconstruction/`, `IBM_Cloud/outputs/hvk2d_cifar_hardware_reconstruction/` |
| §IV-E | Calibrated-noise shot-budget trade-off | `IBM_Cloud/run_hardware_robustness_simulator_sweep.py` | `IBM_Cloud/outputs/` |
| §IV-F | Real-hardware anchor points (2nd/3rd backend) | `IBM_Cloud/run_hardware_robustness_real_anchors.py` | `IBM_Cloud/outputs/` |
| §IV-G, Table on $D_4$ equivariance | $D_4$ equivariance by construction (design-correctness check, not a quantum result) | `Main2/newHVK/run_d4_symmetry_experiment.py`, `Main2/newHVK/run_extended_validation.py` | `Main2/newHVK/results/d4_symmetry_experiment/`, `Main2/newHVK/results/extended_validation/d4_equivariance/` |
| §IV-H | Restricted pair-correlation / entanglement-necessity diagnostic | `Main2/newHVK/run_newhvk_suite.py` (full-ablation-suite mode) | `Main2/newHVK/results/full_ablation_suite/full_ablation_summary.json` |
| §IV-I | Training-dynamics readout + its disqualifying on/off negative control (4/4 vs 4/4) | `Main2/newHVK/run_phase_transition_multi_dataset.py` | `Main2/newHVK/results/phase_transition_multi_dataset/` |
| §V-A, Table X | Hamiltonian and observable-sector controls | historical script lost (see `Main_new/README.md` for the reproducibility-gap writeup); fresh same-environment reproduction + energy-as-decoder-feature follow-up in `Main_new/` (HVK1D) and `Main_new2/` (HVK2D); no-obs-noise/ZZ-only extension: `Main2/newHVK/run_hamiltonian_reproducibility_extension.py` | printed run output, see `Main_new/README.md` tables; `Main2/newHVK/results/hamiltonian_reproducibility_extension/summary.json` |
| §V-D | Topology alignment (HVK1D vs. HVK2D, 0.16 dB) | `Main2/newHVK/run_topology_comparison.py` (real-circuit), `Main2/newHVK/run_topology_comparison_surrogate.py` (surrogate sweep) | `Main2/newHVK/results/topology_comparison/` |
| §V-D2 | HVK as a representative design-pattern instance | text-only argument, no driver script | — |

## `supplementary_study.tex` (additional to the above)

| Section | Result | Script | Output |
|---|---|---|---|
| Held-out CIFAR-10 controls (18.80 / 18.12 dB) | ridge-regression proxy over closed-form features | `Main2/newHVK/run_newhvk_suite.py` (`run_real_cifar_holdout`) | `Main2/newHVK/results/q1_validation/` |
| **TOST equivalence test** (±1 dB margin, 6 of 8 controls equivalent) — the *only* evidence licensing the word "competitive" | | `Main2/newHVK/run_tost_equivalence.py` | `Main2/newHVK/results/q1_validation/tost_equivalence.json` |
| Multi-dataset held-out extension (6 datasets) | | `Main2/newHVK/run_multi_dataset_validation.py` | `Main2/newHVK/results/multi_dataset_validation/` |
| **Dataset-level generalization** (real gradient-trained model, 150/50 split, 3 seeds) | | `Main2/newHVK/run_dataset_level_generalization.py` | `Main2/newHVK/results/dataset_level_generalization/` |
| **Hamiltonian reproducibility note + energy-as-decoder-feature** | | `Main_new/` (HVK1D), `Main_new2/` (HVK2D) | printed run output; see `Main_new/README.md` |
| Core multi-seed ablations, 9 variants at matched 240-step budget, 3 seeds | | `Main2/newHVK/run_core_multiseed_240.py` (rebuilt and committed) | `Main2/newHVK/results/core_multiseed_240/` |
| Topology comparison (HVK1D vs HVK2D) | real-circuit + surrogate | `Main2/newHVK/run_topology_comparison.py`, `Main2/newHVK/run_topology_comparison_surrogate.py` | `Main2/newHVK/results/topology_comparison/` |
| Held-out output-level $D_4$ symmetry evaluation (5 modes, 5 seeds) | | `Main2/newHVK/run_d4_symmetry_experiment.py` | `Main2/newHVK/results/d4_symmetry_experiment/` |
| Replay job ledger (IonQ / `ibm_marrakesh`) — **auditability only, backs no claim** | | `IBM_Cloud/run_checkpoint_bond_dimension_hardware_sweep.py`, `IBM_Cloud/run_checkpoint_bond_dimension_temperature_hardware_sweep.py`, `IBM_Cloud/run_ionq_system_size_shots_sweep.py` | `IBM_Cloud/outputs/checkpoint_bond_dimension_*_hardware_sweep/` |
| Scaling study | | `Main2/newHVK/run_scaling_study.py` | `Main2/newHVK/results/scaling_study/` |

**Withdrawn — do not reintroduce.** The change-point / "critical epoch" / "critical
temperature" detection results, their finite-size ($N=4,6,8$) and bond-dimension
($\chi$) sweeps, and the IBM/IonQ replays *of that diagnostic* were removed from both
manuscripts on 2026-08-13. No sharp transition exists in any trace; the apparent
detection was a within-run median+2σ threshold artifact, and the on/off negative
control fires 4/4 with the Hamiltonian energy identically zero — exactly as often as
4/4 with it on. Their driver scripts (`run_finite_size_phase_transition.py`,
`run_bond_dimension_phase_transition.py`, `run_critical_temperature.py`,
`run_qubit_energy_phase_transition.py`) remain in the tree for provenance, and 27
orphaned figures were deleted. See `project_artifacts/submission_claim_audit.json` →
`withdrawn_claims`. `tests/test_submission_claim_audit.py::test_no_transition_claims_in_manuscripts`
fails the build if transition language reappears without a disclaimer.

## `literature_review.tex`

**Not submitted.** Internal literature synthesis kept for the authors' reference;
no experimental driver script. Its verified references were folded into
`paper_hvk.tex`'s introduction and discussion at the points where they are actually
cited, rather than shipping the review as a standalone companion. Bibliography
entries verified against primary sources (arXiv/journal pages) on 2026-07-28
— see `experiments/todo` section E for the per-entry verification log.

## Known gaps (do not silently treat these tables as reproduced)

1. **Table `hamiltonian_controls`, 1 of 9 rows** — the original generating
   script was never committed and is unrecoverable. A fresh same-environment
   rerun of the legacy/no-energy/contrastive rows reproduces PSNR 6-12 dB
   *higher* across the board, an unexplained reproducibility gap disclosed
   explicitly in `supplementary_study.tex` §"A Reproducibility Note on Table
   hamiltonian_controls". The no-observable-noise and ZZ-only rows were
   rerun at the same protocol and land within 1 dB of published (unlike the
   other three) — also disclosed there, unexplained. Only
   `contrastive+no-energy` (33.33 dB) still has no artifact of any kind.

2. **Shuffle-observables ablation — withdrawn, not reproduced.** Two measurements
   disagree by more than an order of magnitude (an early write-up reported a
   0.301±0.054 dB drop, a later rebuilt verifier ≈16 dB), and the original
   generating script was never committed, so the discrepancy cannot be
   adjudicated. Neither figure is reported as a result; no claim depends on it.
   See `Main2/newHVK/results/ablation_study/legacy_hvk_controls/eval_controls/shuffle-observables/INTERPRETATION.md`
   and the footnote in `supplementary_study.tex` §"A Fresh, Matched-Budget Rerun".

Resolved since the last pass (2026-07-29): Table II's 5 missing datasets
(rerun and merged — see `project_artifacts/submission_claim_audit.json` →
`sameset_multi_dataset`); the HVK1D/HVK2D ansatz figure generator (rewritten
in `IBM_Cloud/generate_ansatz_figures.py`); `run_core_multiseed_240.py` and
`verify_shuffle_permutations.py` (both rebuilt and committed —
`verify_shuffle_permutations.py`'s finding is a genuine, still-unresolved
discrepancy versus the historical number, not a missing script; see
`.../shuffle-observables/INTERPRETATION.md`); `results-core-map.md` (now
exists at `TODO/results-core-map.md`, fully filled in, finer-grained than
this file); zero-shot generalization (rebuilt from scratch, see the table
above — new numbers land within 0.5 dB of the previously unsourced ones);
`hamiltonian_controls`'s no-observable-noise/ZZ-only rows (rebuilt, see
above).

Environment: PennyLane 0.42.3, PyTorch 2.6.0+cu124 (see individual results
directories for any run-specific deviations). Multi-seed studies default to
seeds `{0,1,2,3,4}` unless a section states otherwise (see
`supplementary_study.tex` §Reproducibility for exceptions).
