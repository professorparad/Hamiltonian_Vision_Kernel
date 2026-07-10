# newHVK publication workspace

This folder is an isolated second-generation HVK workspace. It does not overwrite the legacy `Main`, `Main2`, `Baselines`, or `experiments/quantum_contribution` folders.

## Purpose

The original HVK ablations showed that the observable channel matters, but they did not show quantum advantage: no-entanglement, freeze-quantum, no-energy-loss, and classical replacements were too competitive.

`newHVK` therefore adds a stricter diagnostic:

- keep the completed CIFAR, Monalisa, IBM, and legacy ablation evidence;
- add a restricted pair-correlation benchmark where the target explicitly depends on nonlocal feature products;
- compare entangling observables against no-entanglement, parameter-matched classical, and raw-linear controls;
- write a separate paper draft under `paper_latex/`.

## Important claim boundary

This folder is a quantum-advantage **candidate** workspace. It must not be described as a proven hardware quantum advantage result unless the follow-up held-out CIFAR, multi-seed, hardware-noise, and parameter-matched classical tests also remain positive.

## Run

```bash
./main2/newHVK/scripts/run_all.sh
```

Outputs are written to:

- `main2/newHVK/results/quantum_advantage_candidate/`
- `main2/newHVK/results/baselines/`
- `main2/newHVK/results/ablation_study/`
- `main2/newHVK/results/hardware_probe/`
- `main2/newHVK/paper_latex/newhvk_paper.tex`
- `main2/newHVK/paper_latex/newhvk_paper.pdf`

