# HVK1D/HVK2D validation workspace

This folder is an isolated validation workspace for the HVK1D/HVK2D paper. The
directory name is historical; the reported architecture family is HVK1D and
HVK2D, not a separate `newHVK` model. It does not overwrite the legacy `Main`,
`Main2`, `Baselines`, or `experiments/quantum_contribution` folders.

## Purpose

The retained held-out studies do not establish quantum advantage: local/raw and resource-matched classical features remain competitive on ordinary image reconstruction. The synthetic restricted-pair diagnostic separately tests when entangling observables become useful. An earlier Monalisa freeze-isolation aggregate is excluded from manuscript evidence because its per-seed artifacts are unavailable.

This workspace therefore adds stricter diagnostics for HVK1D/HVK2D:

- keep the completed CIFAR, Monalisa, IBM, and legacy ablation evidence;
- add a restricted pair-correlation benchmark where the target explicitly depends on nonlocal feature products;
- compare entangling observables against no-entanglement, parameter-matched classical, raw-linear, random-VQC, freeze-quantum, and freeze-classical controls;
- generate multi-seed summaries, held-out CIFAR-style proxy tests, observable-noise hardware proxies, epoch CSVs, order-parameter diagnostics, plots, GIFs, and MP4 videos;
- generate a Q1-validation layer with real held-out CIFAR splits, strict same-width classical controls, observable/gate ablations, shuffled-pair controls, finite-shot noise simulation, and a compact IEEE-style PDF addendum;
- generate a CIFAR-derived nonlocal patch-correlation diagnostic where the target depends on distant patch products and pair-observable features become load-bearing;
- generate extended reviewer diagnostics, including no-download MedMNIST/MNIST/Fashion-MNIST second-dataset loading when cached data is present;
- keep paper generation optional so auxiliary LaTeX is not recreated unless `--write-paper` is passed.

## Important claim boundary

This folder is a quantum-advantage **candidate** workspace. It must not be described as a proven hardware quantum advantage result unless the follow-up held-out CIFAR, multi-seed, hardware-noise, and parameter-matched classical tests also remain positive.

For a plain-language summary of every new validation layer, what was tested,
what was proved, and what was not proved, see:

```text
main2/newHVK/WHAT_WAS_TESTED_AND_PROVED.md
```

## Run

```bash
./main2/newHVK/scripts/run_all.sh
```

or run only the full ablation/media suite:

```bash
./main2/newHVK/scripts/run_full_newhvk_ablation_suite.sh
```

or run only the stronger Q1-validation suite and PDF addendum:

```bash
./main2/newHVK/scripts/run_q1_validation_suite.sh
```

or run only the CIFAR nonlocal correlation advantage diagnostic:

```bash
./main2/newHVK/scripts/run_cifar_nonlocal_advantage.sh
```

or run the extended reviewer diagnostics for the MedMNIST/MNIST/Fashion-MNIST
second-dataset loader,
matched 1D/2D topology comparison, D4 equivariance, and MPS ordering:

```bash
./main2/newHVK/scripts/run_extended_validation.sh
```

or run the richer multi-dataset validation across CIFAR-10, MNIST,
Fashion-MNIST, PathMNIST, BloodMNIST, PneumoniaMNIST, and Wisconsin Breast
Cancer:

```bash
.venv/bin/python main2/newHVK/run_multi_dataset_validation.py --download --limit 400 --seeds 0 1 2
```

The second-dataset loader does not download data. It first checks local NPZ
files such as `main2/newHVK/datasets/pathmnist.npz`, then recursively checks
`medmnist` and `medminist` folders, then cached MedMNIST roots such as
`$MEDMNIST_ROOT` and `~/.medmnist`, then cached torchvision/Keras MNIST roots.
If no real dataset cache is found, outputs are explicitly labeled
`synthetic-fashion-like-fallback`.

Outputs are written to:

- `main2/newHVK/results/quantum_advantage_candidate/`
- `main2/newHVK/results/full_ablation_suite/full_ablation_summary.csv` (synthetic restricted-pair diagnostic)
- `main2/newHVK/results/full_ablation_suite/multi_seed_results.csv`
- `main2/newHVK/results/full_ablation_suite/heldout_cifar_proxy.csv`
- `main2/newHVK/results/full_ablation_suite/noise_hardware_probe.csv`
- `main2/newHVK/results/full_ablation_suite/hvk_epoch_reconstruction_table.csv`
- `main2/newHVK/results/full_ablation_suite/hvk_epoch_correlation_table.csv`
- `main2/newHVK/results/full_ablation_suite/order_parameter_curve.csv`
- `main2/newHVK/results/full_ablation_suite/*.png`
- `main2/newHVK/results/full_ablation_suite/media/*.gif`
- `main2/newHVK/results/full_ablation_suite/media/*.mp4`
- `main2/newHVK/results/q1_validation/real_cifar_holdout.csv`
- `main2/newHVK/results/q1_validation/real_cifar_holdout_summary.csv`
- `main2/newHVK/results/q1_validation/observable_gate_ablation.csv`
- `main2/newHVK/results/q1_validation/shot_noise_real_cifar.csv`
- `main2/newHVK/results/q1_validation/resource_comparison.csv`
- `main2/newHVK/results/cifar_nonlocal_advantage/cifar_nonlocal_pair_summary.csv`
- `main2/newHVK/results/cifar_nonlocal_advantage/cifar_nonlocal_advantage.png`
- `main2/newHVK/results/extended_validation/`
- `main2/newHVK/results/multi_dataset_validation/`
- `main2/newHVK/paper_latex/newhvk_q1_validation_report.tex`
- `main2/newHVK/paper_latex/newhvk_q1_validation_report.pdf`
- `main2/newHVK/results/baselines/`
- `main2/newHVK/results/ablation_study/`
- `main2/newHVK/results/hardware_probe/`

To regenerate the auxiliary HVK validation LaTeX source explicitly:

```bash
.venv/bin/python main2/newHVK/run_newhvk_suite.py --full-suite --write-paper
```
