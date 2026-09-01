# Results ↔ Code Map — Springer manuscripts

Every table and figure in the canonical Springer sources
(`overleaf_docs/paper_hvk_springer.tex`, `overleaf_docs/supplementary_study.tex`)
mapped to the driver script, the exact command, its key parameters, and the retained
artifact. Numbering is taken from the compiled PDFs (the `.aux` `\newlabel` entries), so
"Table 4" here is the number a reader sees.

This file replaces, for the Springer manuscripts, the section labels used in
`TODO/results-core-map.md`. That file stays as the **historical** map against the old
IEEEtran `paper_hvk.tex` (§IV-A, §V-C…); its `R1`–`R14` row IDs are quoted in the
`Prior ID` column below so the two can be read side by side. No number changed in the
move to Springer — only the section/table/figure labels around them.

Run every command from the repository root unless the row says otherwise.

**Status vocabulary**

| Status | Meaning |
|---|---|
| `backed` | Driver, parameters and artifact all present, and the artifact reproduces the printed number. |
| `backed (device-side pending)` | Same, but a real-hardware row whose job has not yet been re-confirmed against the IBM Quantum account (see §F2 below). |
| `descriptive` | No measurement — a definitions/architecture table with nothing to reproduce. |
| `partial` | Some rows of the item lack an artifact; the gap is disclosed in the manuscript itself. |

**Path case is load-bearing.** Both `main2/newHVK/` and `Main2/newHVK/` exist and are
different directories; copy the paths exactly as written.

---

## A. Main paper — `paper_hvk_springer.tex`

| Item | § | Prior ID | Driver + command | Key parameters | Artifact | Status |
|---|---|---|---|---|---|---|
| Fig. 1 `fig:hvk_ansatz` | 2 | R14 | `python IBM_Cloud/generate_ansatz_figures.py` | `n_qubits=6`, `n_layers=2`; HVK1D = CNOT ring (range 1 then 2), HVK2D = `grid_edges()` (same 7 edges both layers) | `overleaf_docs/assets/figures/hvk1d_ansatz.pdf`, `hvk2d_ansatz.pdf` | `backed` — generator written 2026-07-29; no original existed despite the caption's claim |
| Table 1 `tab:variants` | 2 | — | — | — | — | `descriptive` (variant/topology/latent-dim/symmetry definitions) |
| Table 2 `tab:sameset_multi_dataset` | 4.1 | R1 | `python Main2/newHVK/run_full_dataset_sameset.py --datasets cifar10 mnist fashion-mnist pathmnist --images-per-dataset 3` then the same with `--datasets bloodmnist --images-per-dataset 2` (the script overwrites its output path rather than merging, so batch and merge by hand) | `--images-per-dataset` 3 for cifar10/mnist/fashion-mnist/pathmnist, 2 for bloodmnist/pneumoniamnist (matches the table's own *n* column); `--epochs` 100 default; patch 8×8; χ=4 | `Main2/newHVK/results/full_dataset_sameset/summary.json` | `backed` — all 6 rows fresh-run, exact match (2026-07-29) |
| Table 3 `tab:hardware_pilot_summary` | 4.4 | R3 | `python IBM_Cloud/run_hvk_hardware_reconstruction.py --shots 256` (Monalisa) and `python IBM_Cloud/run_hvk2d_cifar_hardware_reconstruction.py --shots 256` (4 CIFAR) | `--max-patches 16`, `--shots 256`, backend `ibm_fez`; no retraining, decoder unchanged | `IBM_Cloud/outputs/hardware_reconstruction/hardware_reconstruction_report.json`; `IBM_Cloud/outputs/hvk2d_cifar_hardware_reconstruction/summary.json` | `backed (device-side pending)` — recomputed from raw JSON: 25.896→25.90, 31.521→31.52, CIFAR mean 29.564±2.029→29.56±2.03 |
| Fig. 2 `fig:hardware_reconstruction_monalisa` | 4.4 | R3 | same as Table 3 (Monalisa driver) | 16 patches, 3 measurement bases (Z, X, Y) | `IBM_Cloud/outputs/hardware_reconstruction/` | `backed (device-side pending)` |
| Fig. 3 `fig:hardware_reconstruction_cifar` | 4.4 | R3 | same as Table 3 (CIFAR driver) | 4 images × 16 patches, 2 bases (Z, X) | `IBM_Cloud/outputs/hvk2d_cifar_hardware_reconstruction/` | `backed (device-side pending)` |
| Fig. 4 `fig:hardware_robustness_shot_sweep` | 4.5 | R4 | `python IBM_Cloud/run_hardware_robustness_simulator_sweep.py` | shots {256, 512, 1024, 4096}, 3 repeats, 5 checkpoints, `FakeFez` calibrated-noise simulator against the ideal statevector; **no QPU time** | `IBM_Cloud/outputs/hardware_robustness_study/simulator_sweep.json` (65 rows) | `backed` — recomputed: 8.677→8.68 dB; 4096-vs-256 delta −0.046→−0.05 dB |
| Table 4 `tab:hardware_anchors` | 4.6 | R5 | `python IBM_Cloud/run_hardware_robustness_real_anchors.py` | backends `ibm_marrakesh`, `ibm_kingston`; shots 256 / 1024; the same two checkpoints as the pilot (Monalisa, CIFAR cat) | `IBM_Cloud/outputs/hardware_robustness_study/real_hardware_anchors.json` | `backed (device-side pending)` — 25.942→25.94, 26.103→26.10, 28.926→28.93, 31.237→31.24 |
| Table 5 `tab:d4_equivariance` | 4.7 | R6 | `python main2/newHVK/run_extended_validation.py` (lowercase `main2`) | 1000 cached CIFAR images × 7 non-identity transforms = 7000 evaluations | `main2/newHVK/results/extended_validation/d4_equivariance/d4_equivariance_summary.json` | `backed` — 9.573879468686641e-17 → printed 9.57e-17; driver inferred from the output directory (the historical map left it `confirm`) |
| Fig. 5 `fig:d4_equivariance` | 4.7 | R6 | same as Table 5 | same | same | `backed` |
| Table 6 `tab:hvk_pair_diagnostic` | 4.8 | R7 | `python main2/newHVK/run_newhvk_suite.py` (full-ablation-suite mode) | 5 seeds; leakage-audited synthetic distant-product target; fixed linear readout | `main2/newHVK/results/full_ablation_suite/full_ablation_summary.json` | `backed` — 0.9735224984535916 → printed 0.9735 |
| Fig. 6 `fig:hvk_pair_diagnostic` | 4.8 | R7 | same as Table 6 | same | same | `backed` |
| Table 7 `tab:hamiltonian_controls` | 5.1 | R10 | fresh reproduction in `Main_new/` (HVK1D) and `Main_new2/` (HVK2D); the original generating script was never committed | Monalisa 256×256, patch 64, 200 steps, seed 42, `energy_loss_mode=linear` | printed run output — see `Main_new/README.md`; extension rows: `Main2/newHVK/results/hamiltonian_reproducibility_extension/summary.json` | `partial` — the 38.63 / 44.56 / 42.92 dB values are the fresh same-environment measurement the manuscript adopts; the historical values it quotes as *withdrawn* are unreproducible by construction and labelled as such (see §F3) |
| Table 8 `tab:differentiation` | 5.3 | — | — | — | — | `descriptive` (architecture comparison against reference designs) |

---

## B. Supplement — `supplementary_study.tex`

| Item | § | Prior ID | Driver + command | Key parameters | Artifact | Status |
|---|---|---|---|---|---|---|
| Table 1 `tab:core_multiseed_240` | 3.2 | — | `python Main2/newHVK/run_core_multiseed_240.py` | 9 variants × 3 seeds, matched 240-step budget, identical LR/schedule; `scope_reduced: true` recorded in the run's own summary | `Main2/newHVK/results/core_multiseed_240/summary.json`, `summary.csv`, `runs.csv` | `backed` |
| Table 2 `tab:hvk_real_cifar` | 3.3 | R12 | `python main2/newHVK/run_newhvk_suite.py` (`run_real_cifar_holdout`) | 5 random splits × 4 held-out images = 20 reconstructions; feature width 32; readout 2112 params | `main2/newHVK/results/q1_validation/real_cifar_holdout.json`, `real_cifar_holdout_summary.csv` | `backed` |
| Table 3 `tab:hvk_real_cifar_stats` | 3.3 | R12 | same as Table 2 | per-seed image differences averaged before bootstrap / Wilcoxon | `main2/newHVK/results/q1_validation/paired_statistical_tests.csv`, `.json` | `backed` — −0.68 dB, CI [−0.91, −0.46], p=0.0625 |
| Fig. 1 `fig:hvk_real_cifar` | 3.3 | R12 | same as Table 2 | — | `main2/newHVK/results/q1_validation/` | `backed` |
| Table 4 `tab:tost_equivalence` | 3.3.1 | — | `python Main2/newHVK/run_tost_equivalence.py` | δ = ±1 dB margin, α = 0.05, same 5 seed-level paired differences as Table 3 | `Main2/newHVK/results/q1_validation/tost_equivalence.json` | `backed` — 6 of 8 controls equivalent |
| Table 5 `tab:multi_dataset_reconstruction` | 3.3.1 | — | `python main2/newHVK/run_multi_dataset_validation.py` | 6 datasets × 400 cached images, 3 stratified split seeds, identical protocol per feature map | `main2/newHVK/results/multi_dataset_validation/all_image_datasets_summary.csv`, `all_image_datasets_paired_stats.json` | `backed` |
| Table 6 `tab:generalization_controls` | 3.4 | R2 | `python Main2/newHVK/run_zero_shot_generalization.py --epochs 100` | second image = `Main/data/handofgod_micheal_angelo.jpg`; `Quantum2DGridModel` + `PatchDecoder2D`, 8×8 patches, 100 steps/stage, seed 0 | `Main2/newHVK/results/zero_shot_generalization/summary.json` | `backed` — rebuilt 2026-07-29; lands within 0.5 dB of the earlier unsourced figures at both stages |
| Table 7 `tab:dataset_level_generalization` | 3.5 | — | `python Main2/newHVK/run_dataset_level_generalization.py` | 150 train / 50 held-out CIFAR-10 images, 3 seeds, 20 full-batch Adam epochs; parameter-matched `ClassicalLinearControl` trained identically | `Main2/newHVK/results/dataset_level_generalization/dataset_level_generalization_summary.json` | `backed` |
| Fig. 2 `fig:capacity_scaling_sweeps` | 3.6 | — | `python Main2/newHVK/run_scaling_study.py` | Monalisa, single seed; step sweep, qubit-count sweep, χ ∈ {1,2,4,8} at a 200-step budget | `Main2/newHVK/results/scaling_study/scaling_study.json` | `backed` (single-seed by construction — see G3) |
| Table 8 `tab:resource_capacity` | 3.6 | R12 | derived from the Table 2 run | feature dim 32, readout 2112 params, 6/4 split for every row | `main2/newHVK/results/q1_validation/` | `backed` |
| Fig. 3 `fig:cifar_metric_comparison` | 3.6 | — | `python Baselines/cifar10_comparisons/main.py` | 5 native-resolution CIFAR-10 images, same-set fit, all 8 methods | `Baselines/cifar10_comparisons/outputs/cifar32_aggregate_metrics.csv`, `.json` (the figure file is the PDF export of `cifar32_metric_comparison`) | `backed` |
| Table 9 `tab:hamiltonian_reproducibility` | 3.7 | R10 | `python Main2/newHVK/run_hamiltonian_reproducibility_extension.py`, plus the `Main_new/` / `Main_new2/` reruns | Monalisa, 200 steps, seed 42; all eight rows measured in one identical environment | `Main2/newHVK/results/hamiltonian_reproducibility_extension/summary.json`; printed output in `Main_new/README.md` | `partial` — see §F3; the withdrawal is disclosed in §3.7 itself |
| Fig. 4 `fig:restricted_diagnostic_summary` | 3.8 | R7 | `python main2/newHVK/run_newhvk_suite.py` (full-ablation-suite mode) | 5 seeds, full variant set | `main2/newHVK/results/full_ablation_suite/full_ablation_summary.json` | `backed` |
| Table 10 `tab:phase_transition_corrected` | 3.9 | R8 | `python Main2/newHVK/run_phase_transition_multi_dataset.py` | 6 datasets × 2 images × 2 seeds = 24 runs; noise-free evaluation-mode forward pass after every optimizer step | `Main2/newHVK/results/phase_transition_multi_dataset/summary.json` (+ per-dataset `*_eval_order_traces.json`) | `backed` — 16/24 |
| Table 11 `tab:phase_transition_onoff` | 3.9 | — | `python Main2/newHVK/run_phase_transition_onoff_control.py` | Monalisa + CIFAR cat, 2 seeds each, 200 epochs; "off" = classical-replacement variant with energy identically zero | `Main2/newHVK/results/phase_transition_onoff_control/summary.json`, `raw_traces.json` | `backed` — 4/4 vs 4/4, the negative control the manuscript rests its no-transition statement on |
| Fig. 5 `fig:critical_temperature` | 3.9 | R9 | `python Main2/newHVK/run_critical_temperature.py` | HVK1D, 2 CIFAR-10 images (cat, ship/hydrofoil) × 4 seeds | `Main2/newHVK/results/critical_temperature/critical_temperature_cifar10.json` | `backed` — 8 traces; the figure shows all of them, no threshold is marked |
| Table 12 `tab:topology_real_circuit` | 4.1 | — | `python Main2/newHVK/run_topology_comparison.py` | 2 training images, overlapping 8×8 patches at stride 4 (49/image), 90-step budget, 3 seeds; held out on 3 unseen CIFAR-10 classes | `Main2/newHVK/results/topology_comparison/real_circuit_confirmation.json` | `backed` (absolute PSNR is low at this budget — see G2) |
| Fig. 6 `fig:topology_comparison_summary` | 4.2 | — | `python Main2/newHVK/run_topology_comparison_surrogate.py` (+ the real-circuit run above for the left panels) | 5 seeds, stratified splits, CIFAR-10 / Fashion-MNIST / PathMNIST | `Main2/newHVK/results/topology_comparison/surrogate_paired_stats.json`, `surrogate_manifest.json` | `backed` |
| Table 13 `tab:d4_symmetry_experiment` | 5 | — | `python Main2/newHVK/run_d4_symmetry_experiment.py` | 5 seeds, 20 train / 10 held-out images per seed, 5 modes incl. the pooled *classical* baseline; output consistency over all 8 D₄ transforms | `Main2/newHVK/results/d4_symmetry_experiment/d4_symmetry_experiment_summary.json` | `backed` |
| Fig. 7 `fig:d4_symmetry_experiment_summary` | 5 | — | same as Table 13 | same | same | `backed` |
| §5.1 real-circuit D₄ confirmation (in-text numbers) | 5.1 | — | same driver, real-circuit mode | 4 trained HVK2D checkpoints replayed with the real PennyLane forward pass, no retraining | `Main2/newHVK/results/d4_symmetry_experiment/d4_real_circuit_confirmation.json` | `backed` — 0.2777 → 5.39e-8 equivariance error; 4.355e-2 → 1.46e-15 output consistency |
| Fig. 8 `fig:ibm_circuit_summary` | 7.2 | R13 | `python IBM_Cloud/run_ibm_hvk_probe.py --variant both --dry-run` | builds both circuits and records `depth` / `count_ops` without submitting; `optimization_level=1`, `ibm_fez` basis | `IBM_Cloud/outputs/circuits_summary.json` | `backed` — HVK1D depth 18 / cx 10, HVK2D depth 18 / cx 14 |
| Table 14 (unlabeled, §7.4 job identifiers) | 7.4 | R3 | same drivers as Table 3 of the paper | 176 circuits across 5 jobs, `SamplerV2`, 256 shots/circuit | the two hardware-pilot output directories | `backed (device-side pending)` — see §F2 |
| Table 15 `tab:replay_job_ledger` | 7.5 | — | `python IBM_Cloud/run_checkpoint_bond_dimension_hardware_sweep.py`, `python IBM_Cloud/run_checkpoint_bond_dimension_temperature_hardware_sweep.py`, `python IBM_Cloud/run_ionq_system_size_shots_sweep.py` | 16 jobs: IonQ ideal simulator + `ibm_marrakesh`, 256/512/1024 shots | `IBM_Cloud/outputs/checkpoint_bond_dimension_hardware_sweep/`, `IBM_Cloud/outputs/checkpoint_bond_dimension_temperature_hardware_sweep/` | `backed (device-side pending)` — auditability only; no claim rests on these |

---

## F2. Hardware provenance ledger

Every job identifier printed in the manuscripts, with the backend, shot count and PSNR
read back from the retained JSON. **The `Account` column is the open half of F2**: it
needs the IBM Quantum account, which is not configured on this machine (no
`~/.qiskit/qiskit-ibm.json`, no `IBM_QUANTUM_TOKEN`), so nothing below has been
confirmed against the service — only against local artifacts.

### Reconstruction pilot — paper Table 3, supplement §7.4

| Image | Backend | Shots | Job ID | Circuits | PSNR (dB) | Account |
|---|---|---|---|---|---|---|
| Monalisa (HVK1D) | `ibm_fez` | 256 | `d9ecu34inv1c73aq3qt0` | 48 (Z, X, Y) | 25.896 | pending |
| CIFAR cat (HVK2D) | `ibm_fez` | 256 | `d9edfo2neu4c739ob2ig` | 32 (Z, X) | 31.521 | pending |
| CIFAR ship / hydrofoil | `ibm_fez` | 256 | `d9edg5cjeosc73filhng` | 32 (Z, X) | 26.440 | pending |
| CIFAR ship / sea boat | `ibm_fez` | 256 | `d9edgakjeosc73filhug` | 32 (Z, X) | 31.196 | pending |
| CIFAR airplane | `ibm_fez` | 256 | `d9edgdphtsac739e4kdg` | 32 (Z, X) | 29.099 | pending |

### Repeated-execution anchors — paper Table 4

| Image | Backend | Shots | Job ID | Circuits | PSNR (dB) | Account |
|---|---|---|---|---|---|---|
| Monalisa (HVK1D) | `ibm_marrakesh` | 256 | `d9gqm58gk0ls73f219m0` | 48 | 25.942 | pending |
| Monalisa (HVK1D) | `ibm_kingston` | 1024 | `d9gqnk0gk0ls73f21bkg` | 48 | 26.103 | pending |
| CIFAR cat (HVK2D) | `ibm_kingston` | 256 | `d9gqqi0gk0ls73f21glg` | 32 | 28.926 | pending |
| CIFAR cat (HVK2D) | `ibm_kingston` | 1024 | `d9gqr8chonhs73ac3fmg` | 32 | 31.237 | pending |

### Replay ledger — supplement Table 15 (auditability only, no claim rests on these)

| Campaign | Platform | Backend | Shots | Job ID | Account |
|---|---|---|---|---|---|
| Order parameter, cross-platform | IonQ ideal sim. | `ionq_simulator` | 256 | `019f903b-62c0-74db-adec-a33847ae8a1c` | pending |
| Order parameter, cross-platform | IonQ ideal sim. | `ionq_simulator` | 512 | `019f903d-09d4-7548-87f1-2a55185ee9b5` | pending |
| Order parameter, cross-platform | IonQ ideal sim. | `ionq_simulator` | 1024 | `019f903e-cac2-708f-911f-67b2b6d7471c` | pending |
| R_ES, three-basis | IonQ ideal sim. | `ionq_simulator` | 1024 | `019f9046-7ba6-7320-9766-552d88d490bf` | pending |
| Bond-dim. order parameter | IBM hardware | `ibm_marrakesh` | 256 | `d9hqu0jsbqfc73eqgr80` | pending |
| Bond-dim. order parameter | IBM hardware | `ibm_marrakesh` | 512 | `d9hr3d8gk0ls73f3elsg` | pending |
| Bond-dim. order parameter | IBM hardware | `ibm_marrakesh` | 1024 | `d9hr43l0k0jc738il8ug` | pending |
| Bond-dim. order parameter | IonQ ideal sim. | `ionq_simulator` | 256 | `019f9571-21d0-7339-9111-d31a308440a1` | pending |
| Bond-dim. order parameter | IonQ ideal sim. | `ionq_simulator` | 512 | `019f9573-1ed9-768c-9768-465fe75a409b` | pending |
| Bond-dim. order parameter | IonQ ideal sim. | `ionq_simulator` | 1024 | `019f9574-a981-71dd-94b0-9d8fb8968f49` | pending |
| Bond-dim. R_ES | IBM hardware | `ibm_marrakesh` | 256 | `d9hr44shonhs73adgq40` | pending |
| Bond-dim. R_ES | IBM hardware | `ibm_marrakesh` | 512 | `d9hr5u50k0jc738ilbs0` | pending |
| Bond-dim. R_ES | IBM hardware | `ibm_marrakesh` | 1024 | `d9hrb8shonhs73adh8mg` | pending |
| Bond-dim. R_ES | IonQ ideal sim. | `ionq_simulator` | 256 | `019f957f-dfbe-70fc-b6aa-b2398317f049` | pending |
| Bond-dim. R_ES | IonQ ideal sim. | `ionq_simulator` | 512 | `019f9584-4ba3-7224-ab7b-e81e32e6c1e2` | pending |
| Bond-dim. R_ES | IonQ ideal sim. | `ionq_simulator` | 1024 | `019f9586-4eb6-7596-ba47-dc6132c7f943` | pending |

### How to close the `Account` column

With the account configured (`QiskitRuntimeService.save_account(...)`, or
`IBM_QUANTUM_TOKEN` in the environment), run
`python IBM_Cloud/verify_hardware_jobs.py --write-map`. It reads every job ID in this
file, retrieves each one from the service, and rewrites the `Account` column with
`ok` / a mismatch note, filling the instance/CRN line below. The IonQ rows are **not**
IBM jobs and are not retrievable through that service — verify those in the IonQ console
if provenance for them is wanted.

- **Instance / CRN:** _to be recorded when the account check is run._
- **Quota note:** the pilot's meter reading (19 → 35 s of a 600 s monthly allowance) was
  recorded contemporaneously but the raw service response was never serialized, so it is
  an execution-log record, not a recomputable artifact. The manuscript says so in §7.4.

---

## F3. R10 `contrastive+no-energy` (33.33 dB) — closed

The historical map lists this as the one row with no artifact of any kind. **It no longer
needs one: the value is not in either Springer manuscript.** `33.33` does not appear in
`paper_hvk_springer.tex` or `supplementary_study.tex`.

What survives is §3.7's explicit account of the *withdrawal*, which quotes three of the
old table's numbers (`32.24 → 33.30`, contrastive `32.84`) precisely to state that they
are superseded and that three of that table's six rows are not independently
reproducible. Those are cited as history, not as results, and the manuscript is
unambiguous about it — so nothing is claimed on the strength of an artifact that does not
exist. The eight rows actually printed in Table 9 all come from
`hamiltonian_reproducibility_extension/summary.json` and the `Main_new/` reruns.

No artifact needs to be produced and no text needs to be cut. If the withdrawn values are
ever promoted back into a claim, this row reopens.

---

## Coverage

- Main paper: 8 tables + 6 figures — all mapped (2 `descriptive`, 1 `partial`, 4 device-side pending).
- Supplement: 15 tables + 8 figures — all mapped (1 `partial`, 2 device-side pending).
- Open: the `Account` column above (F2), and the two physics-review items that need data
  rather than mapping — G2 (is 11.7 dB in Table 12 above the random-VQC floor at the same
  90-step budget?) and G3 (is the χ non-monotonicity in Fig. 2 multi-seed or explicitly
  single-seed?).
