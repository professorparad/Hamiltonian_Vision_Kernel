# HVK figure-evidence audit

This file records the submission status of every figure under `figures/`.
It prevents exploratory plots from being mistaken for retained manuscript
evidence.

## Submission policy

- The main paper retains only figures that support a distinct headline claim.
- Resource-matched controls, extended statistics, and implementation evidence
  belong in `supplementary_study.tex`.
- Individual-run views and superseded exploratory diagnostics remain archived
  for provenance, but are not submission evidence and must not be cited as
  confirming a claim.
- A plot is promoted only when its source data, protocol, sample size, and claim
  scope are stated in the manuscript or supplement.

## Current inventory

- Main manuscript: 13 figure files.
- Supplement: 13 figure files.
- Archived and uncited: 37 figure files.
- Unique files: 62. The category counts sum to 63 because
  `full_ablation_metric_comparison.pdf` is cited in both documents;
  `ibm_circuit_summary.pdf` is intentionally supplement-only.

The main paper uses the architecture diagrams, IBM reconstruction panels,
checkpoint hardware summaries, calibrated-noise sweep, symmetry validation,
restricted entanglement ablation, finite-size and bond-dimension diagnostics,
and energy-to-entanglement traces. The supplement contains the held-out,
capacity, topology, symmetry, contrastive-control, and hardware-methodology
figures.

## Archived qualitative or per-run duplicates

These are useful for provenance or presentations but duplicate aggregate panels
or tables already retained:

- `ablation_baseline_reconstructions.png`
- `ablation_freeze_classical_reconstructions.png`
- `ablation_random_vqc_reconstructions.png`
- `cifar_cat_hvk1d_order_parameters.png`
- `cifar_cat_hvk1d_reconstruction.png`
- `cifar_cat_hvk2d_order_parameters.png`
- `cifar_cat_hvk2d_reconstruction.png`
- `cifar_frog_hvk1d_order_parameters.png`
- `cifar_frog_hvk1d_reconstruction.png`
- `cifar_frog_hvk2d_order_parameters.png`
- `cifar_frog_hvk2d_reconstruction.png`
- `cifar_frog_symmetric_hvk1d_order_parameters.png`
- `cifar_frog_symmetric_hvk1d_reconstruction.png`
- `monalisa_hvk1d_order_parameters.png`
- `monalisa_hvk2d_order_parameters.png`
- `monalisa_symmetric_hvk1d_order_parameters.png`
- `monalisa_original_vs_reconstruction_epoch_200.png`
- `real_cifar_reconstruction_panel.png`

## Superseded exploratory diagnostics

These plots predate the corrected multi-seed/change-point framing or are
single-run views. They are not evidence of a thermodynamic phase transition:

- `ibm_epoch_proxy_loss_vs_epoch.png`
- `ibm_hvk_probe_metrics.png`
- `mps_bond_dim_scaling_1d.png`
- `mps_bond_dim_scaling_2d.png`
- `multi_dataset_order_parameter_trajectories.pdf`
- `multi_dataset_phase_transition_gallery.pdf`
- `order_parameter_phase_transition_panel.png`
- `phase_transition_ablation_zoom.pdf`
- `phase_transition_cifar_cat_single.pdf`
- `phase_transition_monalisa_single.pdf`
- `phase_transition_single_display.pdf`
- `reconstruction_error_vs_epoch.png`

## Redundant aggregate or scope-limited plots

These results are represented more clearly by retained tables, composite
figures, or the supplement:

- `cifar32_all_architecture_benchmark.png`
- `cifar32_metric_comparison.png`
- `component_ablation_summary.pdf`
- `monalisa_metric_comparison.png`
- `monalisa_original_vs_heron_patch_proxy.png`
- `no_mps_reconstructions.png`
- `second_image_generalization_reconstructions.png`

## IonQ evidence placement

IonQ ideal cloud-simulator results are intentionally summarized in prose in the
main paper. Complete shot budgets, deviations, job identifiers, and the
three-basis energy-to-entanglement replays are reported in
`supplementary_study.tex`. Both bond-dimension hardware figures are
supplement-only: the main paper retains a concise checkpoint-executability result,
while the supplement carries the full curves and denominator-limit analysis.
No IonQ-simulator plot is added to the main paper because it would duplicate the
checkpoint trajectories and could visually blur the distinction between ideal
simulation and QPU hardware.
