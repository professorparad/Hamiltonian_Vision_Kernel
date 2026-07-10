# newHVK publication workspace

This folder is an isolated second-generation HVK workspace. It does not overwrite the legacy `Main`, `Main2`, `Baselines`, or `experiments/quantum_contribution` folders.

## Purpose

The original HVK ablations showed that the observable channel matters, but they did not show quantum advantage: no-entanglement, freeze-quantum, no-energy-loss, and classical replacements were too competitive.

`newHVK` therefore adds a stricter diagnostic:

- keep the completed CIFAR, Monalisa, IBM, and legacy ablation evidence;
- add a restricted pair-correlation benchmark where the target explicitly depends on nonlocal feature products;
- compare entangling observables against no-entanglement, parameter-matched classical, raw-linear, random-VQC, freeze-quantum, and freeze-classical controls;
- generate multi-seed summaries, held-out CIFAR-style proxy tests, observable-noise hardware proxies, epoch CSVs, order-parameter diagnostics, plots, GIFs, and MP4 videos;
- keep paper generation optional so the deleted `newhvk_paper.tex` is not recreated unless `--write-paper` is passed.

## Important claim boundary

This folder is a quantum-advantage **candidate** workspace. It must not be described as a proven hardware quantum advantage result unless the follow-up held-out CIFAR, multi-seed, hardware-noise, and parameter-matched classical tests also remain positive.

## Run

```bash
./main2/newHVK/scripts/run_all.sh
```

or run only the full ablation/media suite:

```bash
./main2/newHVK/scripts/run_full_newhvk_ablation_suite.sh
```

Outputs are written to:

- `main2/newHVK/results/quantum_advantage_candidate/`
- `main2/newHVK/results/full_ablation_suite/full_ablation_summary.csv`
- `main2/newHVK/results/full_ablation_suite/multi_seed_results.csv`
- `main2/newHVK/results/full_ablation_suite/heldout_cifar_proxy.csv`
- `main2/newHVK/results/full_ablation_suite/noise_hardware_probe.csv`
- `main2/newHVK/results/full_ablation_suite/hvk_epoch_reconstruction_table.csv`
- `main2/newHVK/results/full_ablation_suite/hvk_epoch_correlation_table.csv`
- `main2/newHVK/results/full_ablation_suite/order_parameter_curve.csv`
- `main2/newHVK/results/full_ablation_suite/*.png`
- `main2/newHVK/results/full_ablation_suite/media/*.gif`
- `main2/newHVK/results/full_ablation_suite/media/*.mp4`
- `main2/newHVK/results/baselines/`
- `main2/newHVK/results/ablation_study/`
- `main2/newHVK/results/hardware_probe/`

To regenerate the deleted newHVK LaTeX source explicitly:

```bash
.venv/bin/python main2/newHVK/run_newhvk_suite.py --full-suite --write-paper
```
