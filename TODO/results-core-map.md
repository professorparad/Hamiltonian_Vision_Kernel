# Results ↔ Code Map

Every number, table, and figure in `paper_hvk.tex` must map to the exact script,
its parameters, and the result artifact. Replace every `<<<FILL>>>` with the real
value. Guessed script names are marked "confirm". A row with no artifact does not
ship — fix it or cut the claim.

Status values: `backed` (code + params + artifact all present and match the paper),
`confirm` (artifact exists, provenance unverified), `no source` (must fix or cut).

| ID | Paper element | Reported value | Script + command | Key parameters | Artifact | Status |
|----|---------------|----------------|------------------|----------------|----------|--------|
| R1 | Table `sameset_multi_dataset` (§IV-A) | 26.6–41.6 dB, SSIM 0.89–1.00, 6 datasets | `run_full_dataset_sameset.py` (confirm) — `<<<FILL>>>` | steps=`<<<FILL>>>` seeds=`<<<FILL>>>` patch=8×8 χ=4 lr=`<<<FILL>>>` | `main2/newHVK/results/full_dataset_sameset/summary.json` | no source — only PneumoniaMNIST present; regenerate 6 or cut |
| R2 | Zero-shot transfer (§IV-B) | 7.78 dB → 28.31 dB | `<<<FILL>>>` | `<<<FILL>>>` | `<<<FILL>>>` | no source |
| R3 | Table `hardware_pilot_summary` + Figs (§IV-C) | HW 25.90–31.52 dB; CIFAR 29.56±2.03 vs sim 42.69±3.84 | `run_hvk_hardware_reconstruction.py`, `run_hvk2d_cifar_hardware_reconstruction.py` — `<<<FILL>>>` | backend=ibm_fez, shots=256/basis, ckpt=`<<<FILL>>>` | `IBM_Cloud/outputs/hardware_reconstruction/hardware_reconstruction_report.json`; `IBM_Cloud/outputs/hvk2d_cifar_hardware_reconstruction/summary.json` | backed (job IDs) |
| R4 | Fig `hardware_robustness_shot_sweep` (§IV-D) | 8.68 dB drop at 256 shots; +shots −0.05 dB avg | `run_hardware_robustness_simulator_sweep.py` (confirm) — `<<<FILL>>>` | shots={256,512,1024,4096}, reps=3 | `IBM_Cloud/outputs/hardware_robustness_study/simulator_sweep.json` | confirm aggregates recompute |
| R5 | Table `hardware_anchors` (§IV-E) | marrakesh 25.94; kingston 26.10 / 28.93 / 31.24 | `run_hardware_robustness_real_anchors.py` (confirm) — `<<<FILL>>>` | backends=marrakesh,kingston, shots=256/1024 | `IBM_Cloud/outputs/hardware_robustness_study/real_hardware_anchors.json` | backed (job IDs) |
| R6 | Table + Fig `d4_equivariance` (§V-A) | pooled 9.57e-17; unpooled 0.74–0.84 | `run_d4_symmetry_experiment.py` / `run_extended_validation.py` (confirm) — `<<<FILL>>>` | n_images=1000, transforms=7 | `main2/newHVK/results/extended_validation/d4_equivariance/d4_equivariance_summary.json` | backed |
| R7 | Table + Fig `hvk_pair_diagnostic` (§V-B) | entangling R²=0.9735 vs ≤0.02 controls | `run_newhvk_suite.py` (confirm) — `<<<FILL>>>` | seeds=5, steps=`<<<FILL>>>`, lr=`<<<FILL>>>`, target-seed=`<<<FILL>>>` | `main2/newHVK/results/full_ablation_suite/full_ablation_summary.json` | backed (leakage-audited) |
| R8 | Table `phase_transition_corrected` (§V-C) | 16/24 detected | `run_phase_transition_multi_dataset.py` (confirm) — `<<<FILL>>>` | 2 img × 2 seeds/dataset, eval-mode trace | `main2/newHVK/results/phase_transition_multi_dataset/summary.json` | backed |
| R9 | Fig `critical_temperature` (§V-D) | 2/4 cross threshold; t_c=125/175 | `run_critical_temperature.py` (confirm) — `<<<FILL>>>` | HVK1D, 2 CIFAR img × 2 seeds | `main2/newHVK/results/critical_temperature/critical_temperature_cifar10.json` | confirm values recompute |
| R10 | Table `hamiltonian_controls` (§VI-A) | baseline 32.24; no-energy 33.30; contrastive 32.84/33.33; ZZ-only 32.88; no-noise 32.72 | `<<<FILL>>>` | single-seed, steps=`<<<FILL>>>`, lr=`<<<FILL>>>`, λ=`<<<FILL>>>` | baseline: `main2/newHVK/results/ablation_study/legacy_hvk_controls/eval_controls/shared-baseline-seed-42/metrics.json`; other 5 rows: `<<<FILL>>>` | partial — only baseline located |
| R11 | Topology Monalisa 1D vs 2D (§V-D text) | 40.70 vs 34.72 dB (5.98 dB) | `<<<FILL>>>` | `<<<FILL>>>` | `<<<FILL>>>` — `topology_comparison/` has no Monalisa run | no source — produce or delete |
| R12 | Held-out CIFAR (Discussion §VI-C) | 18.80±1.42 vs 18.12±1.54; −0.68 dB; Wilcoxon p=0.0625, n=5 | `<<<FILL>>>` (Q1-validation runner) | 5 seeds × 4 held-out img, width 32, readout 2112 | `main2/newHVK/results/q1_validation/real_cifar_holdout_summary.csv`; `.../paired_statistical_tests.csv` | backed |
| R13 | Table `ibm_probe` + Fig (Appendix) | depth-18, 6 qubits, CNOT 10/14 | `run_ibm_hvk_probe.py` (confirm) — `<<<FILL>>>` | `<<<FILL>>>` | `<<<FILL>>>` | confirm artifact |
| R14 | Figs `hvk_ansatz` (§II) | HVK1D / HVK2D circuit diagrams | `<<<FILL>>>` (generator) | — | `figures/hvk1d_ansatz.pdf`; `figures/hvk2d_ansatz.pdf` | confirm generator |

Done when every row is `backed`. Then update `submission_claim_audit.json` to match (after R1 and R11 are resolved).
