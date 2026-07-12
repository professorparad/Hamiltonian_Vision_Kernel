# What Was Added, Tested, and Proved

This note summarizes the new validation work added for the HVK paper. It is
written to keep the claim boundary clear for journal submission.

## Short Answer

The current evidence does **not** prove quantum advantage.

What it does prove is narrower and more defensible:

- The HVK observable channel is load-bearing in the reconstruction pipeline.
- A deliberately constructed pair-observable feature map can solve restricted
  nonlocal correlation diagnostics that local/single-site controls cannot solve
  under the same readout protocol.
- A D4-pooled HVK2D observable-grid map is equivariant to square rotations and
  reflections by construction and by numerical test.
- On ordinary held-out image reconstruction, local/raw classical controls remain
  stronger than the current HVK2D feature map.

So the Q1-safe framing is:

> HVK is a rigorously ablated observable-latent diagnostic framework with a
> D4-equivariant extension and controlled nonlocal-correlation advantages, but
> it does not establish broad natural-image reconstruction or hardware quantum
> advantage.

## New Things Added

### 1. Q1 Held-Out CIFAR Validation

Location:

- `main2/newHVK/results/q1_validation/`
- `main2/newHVK/paper_latex/newhvk_q1_validation_report.tex`
- `main2/newHVK/paper_latex/newhvk_q1_validation_report.pdf`

What was tested:

- HVK2D real-CIFAR map
- no-entanglement control
- ZZ-only control
- local-observables-only control
- raw-linear classical control
- quadratic classical control
- strict same-width random Fourier classical control
- shuffled pair-observable control
- random VQC control

What was found:

- Local/raw controls are best on held-out CIFAR reconstruction.
- HVK2D beats random VQC and strict random features, but does not beat local/raw
  controls.

Main result:

- Local/raw PSNR: about `20.66 ± 1.46 dB`
- HVK2D real-CIFAR PSNR: about `20.07 ± 1.45 dB`

Claim:

- This is a negative result for broad CIFAR reconstruction advantage.
- It strengthens the paper because it prevents overclaiming.

### 2. Paired Statistical Tests

Location:

- `main2/newHVK/results/q1_validation/paired_statistical_tests.csv`
- `main2/newHVK/results/q1_validation/paired_statistical_tests.json`

What was tested:

- HVK2D real-CIFAR map versus each control, paired by seed and held-out image.
- Wilcoxon signed-rank tests.
- Bootstrap 95% confidence intervals for paired PSNR differences.

What was found:

- HVK2D is significantly worse than raw-linear/local controls on held-out CIFAR.
- HVK2D is significantly better than random VQC and strict random features.
- HVK2D is statistically similar to no-entanglement, shuffled-pair, and
  quadratic controls in this small held-out CIFAR setup.

Claim:

- The statistics support a careful diagnostic paper, not a quantum-advantage
  claim.

### 3. D4 Symmetry / Equivariance Diagnostic

Location:

- `main2/newHVK/results/extended_validation/d4_equivariance/`

What was tested:

- Unpooled HVK2D positional map
- no-positional map
- local/raw map
- D4-pooled HVK2D map

What was found:

- The original unpooled positional HVK2D map is **not** D4-equivariant.
- The D4-pooled HVK2D map is equivariant up to numerical precision.

Main result:

- D4-pooled HVK2D mean equivariance error: about `1.01e-16`
- HVK2D positional mean equivariance error: about `7.89e-1`

Claim:

- The D4-pooled extension has a real symmetry guarantee.
- The original HVK2D map should only be described as D4-motivated, not
  D4-equivariant.

### 4. CIFAR Nonlocal Patch-Correlation Diagnostic

Location:

- `main2/newHVK/results/cifar_nonlocal_advantage/`

What was tested:

- A CIFAR-derived task where the target explicitly depends on distant patch
  product observables.
- HVK2D pair-observable features against local, single-site, raw-linear,
  random-feature, random-VQC, and explicit quadratic controls.

What was found:

- The pair-observable map does extremely well when the task is constructed from
  pair products.
- This supports the usefulness of pair-observable features on nonlocal
  correlation tasks.

Claim:

- This is a controlled representational diagnostic.
- It is not proof of general image reconstruction advantage.

### 5. Cached PathMNIST / MedMNIST-Style Second Dataset

Location:

- `main2/newHVK/datasets/pathmnist.npz`
- `main2/newHVK/results/extended_validation/second_dataset_subset/`

Question: did it check a medical dataset?

Answer: **Yes.** The run used the local file:

```text
main2/newHVK/datasets/pathmnist.npz
```

The generated CSV labels it as:

```text
local-npz:/home/adminpc/Desktop/HVK/Script/Hamiltonian_Vision_Kernel/main2/newHVK/datasets/pathmnist.npz
```

What was tested:

- HVK2D pair-observable reconstruction
- no-entanglement
- local-only
- raw-linear
- parameter-matched
- quadratic-classical

What was found:

- Local-only and raw-linear controls were best.
- HVK2D pair-observable was close to quadratic-classical but weaker than
  local/raw controls.

Main result:

- Local/raw PSNR: about `26.85 ± 6.03 dB`
- HVK2D pair-observable PSNR: about `26.25 ± 5.70 dB`

Claim:

- This is a real second-dataset diagnostic from a local PathMNIST-style NPZ.
- It does not prove image reconstruction advantage.

### 6. Wisconsin Breast Cancer Statistics

Location:

- `main2/newHVK/results/extended_validation/wisconsin_breast_cancer/`

Question: did it check Wisconsin statistics?

Answer: **Yes.** The run used the scikit-learn Wisconsin Breast Cancer dataset.

What was tested:

- HVK2D pair-observable features
- no-entanglement features
- raw-linear features
- quadratic-classical features
- strict classical random features

Task:

- Binary classification, not image reconstruction.

What was found:

- Raw-linear was best.
- HVK2D pair-observable was competitive but weaker than raw-linear.

Main result:

- Raw-linear accuracy: about `0.980 ± 0.011`
- HVK2D pair-observable accuracy: about `0.943 ± 0.016`

Claim:

- This is a cross-domain sanity check.
- It should not be cited as image-reconstruction evidence.
- It does not support quantum advantage.

### 7. Rich Multi-Dataset Validation

Location:

- `main2/newHVK/run_multi_dataset_validation.py`
- `main2/newHVK/results/multi_dataset_validation/`

Question: did it check MNIST, Fashion-MNIST, and more medical datasets?

Answer: **Yes.** A later approved network run downloaded and evaluated:

- CIFAR-10 native 32x32: 400 loaded images, 10 classes
- MNIST: 400 loaded images, 10 classes
- Fashion-MNIST: 400 loaded images, 10 classes
- PathMNIST: 400 loaded images, 9 classes
- BloodMNIST: 400 loaded images, 8 classes
- PneumoniaMNIST: 400 loaded images, 2 classes
- Wisconsin Breast Cancer: tabular classification diagnostic

Downloaded/local dataset files include:

```text
main2/newHVK/datasets/MNIST/
main2/newHVK/datasets/FashionMNIST/
main2/newHVK/datasets/pathmnist.npz
main2/newHVK/datasets/bloodmnist.npz
main2/newHVK/datasets/pneumoniamnist.npz
```

What was found:

- Across CIFAR-10, MNIST, Fashion-MNIST, PathMNIST, BloodMNIST, and
  PneumoniaMNIST reconstruction, local/raw controls remain strongest.
- HVK2D pair-observable features remain useful diagnostics but do not beat
  local/raw controls on ordinary reconstruction.
- Paired Wilcoxon/bootstrap tests were added for the multi-dataset
  reconstruction results in:

```text
main2/newHVK/results/multi_dataset_validation/all_image_datasets_paired_stats.csv
```

- A downstream image-classification diagnostic was added in:

```text
main2/newHVK/results/multi_dataset_validation/all_image_classification_summary.csv
```

Representative PSNR results:

- CIFAR-10: local/raw `21.17 dB`, HVK2D pair `19.99 dB`
- MNIST: no-entanglement `16.55 dB`, local/raw `16.54 dB`, HVK2D pair `16.12 dB`
- Fashion-MNIST: local/raw `17.67 dB`, HVK2D pair `17.33 dB`
- PathMNIST: local/raw `27.52 dB`, HVK2D pair `27.01 dB`
- BloodMNIST: local/raw `24.79 dB`, HVK2D pair `24.24 dB`
- PneumoniaMNIST: local/raw `26.49 dB`, HVK2D pair `25.73 dB`

Representative classification accuracy results:

- MNIST: raw-linear `0.663`, HVK2D pair `0.643`
- Fashion-MNIST: raw-linear `0.637`, HVK2D pair `0.577`
- PathMNIST: raw-linear `0.638`, HVK2D pair `0.608`
- BloodMNIST: raw-linear `0.667`, HVK2D pair `0.539`
- PneumoniaMNIST: raw-linear `0.817`, HVK2D pair `0.783`

Claim:

- The multi-dataset evidence makes the paper richer and more credible.
- It also reinforces the negative conclusion: there is no broad reconstruction
  quantum advantage in the current feature map.

## New Scripts / Commands

Run Q1 validation:

```bash
.venv/bin/python main2/newHVK/run_newhvk_suite.py --q1-validation --write-q1-report
```

Run full ablation and Q1 validation:

```bash
.venv/bin/python main2/newHVK/run_newhvk_suite.py --full-suite --q1-validation --write-q1-report --cifar-nonlocal-advantage
```

Run extended reviewer diagnostics:

```bash
.venv/bin/python main2/newHVK/run_extended_validation.py
```

Compile the main paper:

```bash
cd latex_outputs/paper_latex
pdflatex -interaction=nonstopmode -halt-on-error paper_hvk.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_hvk.tex
```

Compile the Q1 addendum:

```bash
cd main2/newHVK/paper_latex
pdflatex -interaction=nonstopmode -halt-on-error newhvk_q1_validation_report.tex
pdflatex -interaction=nonstopmode -halt-on-error newhvk_q1_validation_report.tex
```

## Final Submission Boundary

Safe claim:

> HVK provides a rigorously ablated observable-latent framework with a
> D4-equivariant pooled extension and controlled nonlocal-correlation diagnostic
> advantages.

Unsafe claim:

> HVK proves quantum advantage for natural-image reconstruction.

The unsafe claim is not supported by the current held-out CIFAR, PathMNIST, or
Wisconsin results.
