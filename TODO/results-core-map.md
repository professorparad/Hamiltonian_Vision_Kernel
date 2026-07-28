# Results ↔ Code Map (fill-in)

**Purpose:** every number, table, and figure in `paper_hvk.tex` must map to
(1) the exact script + command that produced it, (2) the key parameters, and
(3) the result artifact file in the repo. If a row cannot be filled, that result
does not ship.

**Student:** fill every `FILL:` field. `Script` = confirm the guessed path and
give the exact run command. `Params` = seeds / steps / patch / χ / lr / shots as
relevant. Set `Status` to ✅ once code + params + artifact all exist and match the
paper number.

Legend — ✅ backed & verified · ⚠ partial · ❌ no source (fix or cut).

---

### R1 — Table `tab:sameset_multi_dataset` (§IV-A, 6-dataset reconstruction)
- **Paper value:** PSNR 26.6–41.6 dB, SSIM 0.89–1.00 across CIFAR/MNIST/Fashion/Path/Blood/Pneumonia
- **Script (guess, confirm):** `main2/newHVK/run_full_dataset_sameset.py`
- **Command:** FILL:
- **Params:** FILL: steps=__ seeds=__ patch=8×8 χ=4 lr=__
- **Artifact:** `main2/newHVK/results/full_dataset_sameset/summary.json`
- **Status:** ❌ only PneumoniaMNIST is in the summary; **regenerate all 6 into one file, or cut the 5 rows**

### R2 — Zero-shot / second-image transfer (§IV-B)
- **Paper value:** 7.78 dB zero-shot; 28.31 dB after adding 2nd image
- **Script:** FILL:
- **Command:** FILL:
- **Params:** FILL:
- **Artifact:** FILL: (no clean source found)
- **Status:** ❌ point to a file or re-run

### R3 — Table `tab:hardware_pilot_summary` + Figs `fig:hardware_reconstruction_{monalisa,cifar}` (§IV-C)
- **Paper value:** HW 25.90–31.52 dB; CIFAR mean 29.56±2.03 vs sim 42.69±3.84
- **Script (confirm):** `IBM_Cloud/run_hvk_hardware_reconstruction.py`, `IBM_Cloud/run_hvk2d_cifar_hardware_reconstruction.py`
- **Command:** FILL:
- **Params:** backend=ibm_fez shots=256/basis; FILL: checkpoint paths
- **Artifact:** `IBM_Cloud/outputs/hardware_reconstruction/hardware_reconstruction_report.json`, `IBM_Cloud/outputs/hvk2d_cifar_hardware_reconstruction/summary.json`
- **Status:** ✅ verified, job IDs present

### R4 — Fig `fig:hardware_robustness_shot_sweep` (§IV-D, FakeFez shot sweep)
- **Paper value:** 8.68 dB noiseless→noisy at 256 shots; +shots changes −0.05 dB avg
- **Script (confirm):** `IBM_Cloud/run_hardware_robustness_simulator_sweep.py`
- **Command:** FILL:
- **Params:** shots={256,512,1024,4096} reps=3; FILL:
- **Artifact:** `IBM_Cloud/outputs/hardware_robustness_study/simulator_sweep.json`
- **Status:** ⚠ file exists — confirm the 8.68 / 4.24 / −0.05 aggregates recompute from it

### R5 — Table `tab:hardware_anchors` (§IV-E, 2nd/3rd backends)
- **Paper value:** marrakesh 25.94; kingston 26.10 / 28.93 / 31.24
- **Script (confirm):** `IBM_Cloud/run_hardware_robustness_real_anchors.py`
- **Command:** FILL:
- **Params:** backends=marrakesh,kingston shots=256/1024; FILL:
- **Artifact:** `IBM_Cloud/outputs/hardware_robustness_study/real_hardware_anchors.json`
- **Status:** ✅ verified, job IDs present

### R6 — Table `tab:d4_equivariance` + Fig `fig:d4_equivariance` (§V-A)
- **Paper value:** pooled error 9.57e-17; unpooled 0.74–0.84
- **Script (confirm):** `main2/newHVK/run_d4_symmetry_experiment.py` / `run_extended_validation.py`
- **Command:** FILL:
- **Params:** n_images=1000 transforms=7 (7000 evals); FILL:
- **Artifact:** `main2/newHVK/results/extended_validation/d4_equivariance/d4_equivariance_summary.json`
- **Status:** ✅ verified

### R7 — Table `tab:hvk_pair_diagnostic` + Fig `fig:hvk_pair_diagnostic` (§V-B, entanglement)
- **Paper value:** entangling R²=0.9735 vs ≤0.02 controls; 7 rows
- **Script (confirm):** `main2/newHVK/run_newhvk_suite.py` (full ablation suite)
- **Command:** FILL:
- **Params:** seeds=5; FILL: steps / lr / target-construction seed
- **Artifact:** `main2/newHVK/results/full_ablation_suite/full_ablation_summary.json`
- **Status:** ✅ verified (leakage-audited target)

### R8 — Table `tab:phase_transition_corrected` (§V-C, phase transition)
- **Paper value:** 16/24 detected; per-dataset t_c, X_max
- **Script (confirm):** `main2/newHVK/run_phase_transition_multi_dataset.py`
- **Command:** FILL:
- **Params:** 2 images × 2 seeds/dataset, eval-mode noise-free trace; FILL:
- **Artifact:** `main2/newHVK/results/phase_transition_multi_dataset/summary.json`
- **Status:** ✅ verified

### R9 — Fig `fig:critical_temperature` (§V-D, effective temperature)
- **Paper value:** 2/4 traces cross threshold; t_c=125/175
- **Script (confirm):** `main2/newHVK/run_critical_temperature.py`
- **Command:** FILL:
- **Params:** HVK1D, 2 CIFAR images × 2 seeds; FILL:
- **Artifact:** `main2/newHVK/results/critical_temperature/critical_temperature_cifar10.json`
- **Status:** ⚠ confirm the detected-count and t_c values recompute from the artifact

### R10 — Table `tab:hamiltonian_controls` (§VI-A, energy/observable controls)
- **Paper value:** baseline 32.24; no-energy 33.30; contrastive 32.84/33.33; ZZ-only 32.88; no-noise 32.72
- **Script:** FILL: (one runner per control, or a sweep)
- **Command:** FILL:
- **Params:** single-seed; FILL: steps / lr / λ
- **Artifact:** baseline in `main2/newHVK/results/ablation_study/legacy_hvk_controls/eval_controls/shared-baseline-seed-42/metrics.json`; **FILL: the other 5 control dirs**
- **Status:** ⚠ only baseline (32.24) located; map the other 5 rows to their metrics.json files

### R11 — Topology 1D vs 2D on Monalisa (§V-D text, "40.70 vs 34.72 dB")
- **Paper value:** 2D exceeds 1D by 5.98 dB on Monalisa
- **Script:** FILL:
- **Command:** FILL:
- **Params:** FILL:
- **Artifact:** FILL: (topology_comparison/ has CIFAR/Fashion/Path only — no Monalisa)
- **Status:** ❌ no source anywhere; **produce the run or delete the sentence**

### R12 — Held-out CIFAR comparison (§VI-C / Discussion)
- **Paper value:** local/raw 18.80±1.42 vs HVK2D 18.12±1.54; diff −0.68 dB; Wilcoxon p=0.0625 n=5
- **Script (confirm):** `main2/newHVK/` Q1-validation runner
- **Command:** FILL:
- **Params:** 5 seeds × 4 held-out images; feature width 32, readout 2112; FILL:
- **Artifact:** `main2/newHVK/results/q1_validation/real_cifar_holdout_summary.csv`, `.../paired_statistical_tests.csv`
- **Status:** ✅ verified

### R13 — Table `tab:ibm_probe` + Fig `fig:ibm_circuit_summary` (Appendix, circuit resources)
- **Paper value:** HVK1D/2D depth-18, 6 qubits, CNOT 10/14
- **Script (confirm):** `IBM_Cloud/run_ibm_hvk_probe.py`
- **Command:** FILL:
- **Params:** FILL:
- **Artifact:** FILL: (locate the probe output json)
- **Status:** ⚠ confirm artifact

### R14 — Figs `fig:hvk_ansatz` (circuit diagrams, §II)
- **Script:** FILL: (diagram generator)
- **Artifact:** `figures/hvk1d_ansatz.pdf`, `figures/hvk2d_ansatz.pdf`
- **Status:** ⚠ confirm generator / provenance

---

**When done:** every row is ✅, and each ✅ has a script + command + params + artifact.
The `FILL:` count is the work remaining. Update `submission_claim_audit.json` to
match once R1 and R11 are resolved.
