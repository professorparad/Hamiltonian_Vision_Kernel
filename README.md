# Hamiltonian Vision Kernel

This repository is a working research codebase for image reconstruction with
Hamiltonian Vision Kernel models. The central idea is to treat image patches as
structured local states, encode those patches with tensor-network features and
position information, then train a small hybrid quantum/classical decoder with a
Hamiltonian-style energy term.

The project is not a polished production library yet. It is closer to a lab
notebook that has been made runnable: scripts, benchmark tables, generated
figures, paper drafts, and a Python package copy live side by side. The code is
useful for reproducing experiments and comparing variants, but some paths are
still deliberately conservative.

## What HVK Is Trying To Test

The question behind the project is simple enough:

Can a reconstruction model use quantum-inspired latent structure, positional
dependence, and Hamiltonian energy terms in a way that behaves like a practical
autoencoder rather than just a toy classifier?

The current pipeline uses:

1. Image patching, so each image is broken into smaller local regions.
2. MPS-style feature extraction, using tensor-network compression features from
   each patch.
3. Sinusoidal positional encoding, so the model knows where the patch came from.
4. A variational quantum circuit, which returns observables such as Z, X, and
   pairwise correlations.
5. A Hamiltonian energy term, used beside reconstruction loss.
6. A small neural decoder, which maps quantum observables back into image
   patches.

The novelty is not in MPS, positional encoding, or Heisenberg-style Hamiltonians
by themselves. Those are known tools. The experiment is in putting them together
for reconstruction and then tracking whether order parameters show useful
training structure.

## Repository Map

- `Main/`: the main HVK1D implementation, including the standard and
  U(1)-symmetric variants.
- `Main2/`: the 2D-grid HVK variant, with lattice-style correlations.
- `Baselines/`: CIFAR-10 and Mona Lisa comparison runners for HVK and classical
  baselines, including CNN, MLP, autoencoder, GAN, PHL, HVK1D, HVK2D, and
  Symmetric HVK1D.
- `python_library/`: package-style copy of the HVK API.
- `IBM_Cloud/`: IBM Quantum hardware probe scripts for Heron-style patch
  circuits. This is not full HVK training.
- `tests/`: unit and smoke tests for preprocessing, decoder behavior, training,
  and benchmark helpers.
- `docs/`: static GitHub Pages site with real paper, metric, and figure
  artifacts.
- `latex_outputs/`: paper drafts, compiled PDF, figures, and exported LaTeX
  material.

Generated outputs are intentionally kept out of git where possible. The virtual
environment folders are ignored as well.

## Environment Setup

On Windows, this repo currently works best with a local Python 3.10 environment:

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA-enabled Torch baselines, install the matching PyTorch CUDA wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python cuda_check.py
```

Expected output:

```text
cuda
```

## Important CUDA Note

Torch CUDA and PennyLane CUDA are different things.

The CNN, MLP, autoencoder, GAN, and PHL baselines can use CUDA through PyTorch.
The HVK quantum circuits require PennyLane's `lightning.gpu` backend if you want
the quantum simulation itself to run on GPU. Native Windows pip currently fails
for that backend in this environment because `custatevec-cu12` is not available.

So on Windows:

- PyTorch baselines: CUDA works.
- HVK quantum circuits: use CPU or `lightning.qubit` unless you move to a Linux
  environment with `pennylane-lightning-gpu`.

For full HVK GPU experiments, WSL2 Ubuntu is the more realistic path.

## Common Commands

Run the main HVK1D pipeline:

```powershell
python Main\main.py --device auto
```

Run CIFAR baselines that can use PyTorch CUDA:

```powershell
python Baselines\cifar10_comparisons\main.py --methods cnn mlp autoencoder gan phl --count 5 --epochs 200 --device cuda
```

Run HVK CIFAR variants safely:

```powershell
python Baselines\cifar10_comparisons\main.py --methods hvk1d hvk2d symmetric --count 5 --epochs 200 --device auto
```

Run the Mona Lisa comparison suite:

```powershell
python Baselines\monalisa_comparisons\main.py --methods all --epochs 200 --device auto
```

Build the current paper PDF:

```powershell
cd latex_outputs\paper_latex
pdflatex -interaction=nonstopmode paper_hvk.tex
pdflatex -interaction=nonstopmode paper_hvk.tex
```

Run a fast smoke test:

```powershell
python Baselines\cifar10_comparisons\smoke_test.py --epochs 1 --methods cnn hvk1d --device auto
```

## Outputs

Most experiment scripts write to `outputs/` folders next to the runner. The
benchmark aggregator also copies results into:

- `outputs/*_per_image_metrics.csv`
- `outputs/*_aggregate_metrics.csv`
- `outputs/*_aggregate_metrics.json`
- `outputs/*_metric_comparison.png`
- `outputs/visuals/<method>/`
- `outputs/per_method_metrics/<method>/`

Those files are meant for analysis, plots, and paper figures. They are not
required to import or run the code.

The current manuscript adds:

- Symmetric HVK1D math and benchmark results.
- CIFAR-10 aggregate comparison across HVK, classical, and PHL baselines.
- Mona Lisa comparison table including CNN, MLP, autoencoder, GAN, PHL, HVK1D,
  HVK2D, and Symmetric HVK1D.
- A provenance audit that excludes exploratory order-parameter trajectories
  until deterministic, evaluation-mode multi-seed traces are available.
- A scoped five-image IBM Heron reconstruction pilot using hardware-measured
  observables and the unchanged trained decoder, with simulator-translation
  checks and complete job/quota metadata in the supplementary study.

The deployed static site is served from `docs/` by `.github/workflows/pages.yml`.
It only links to files committed under `docs/assets`.

## Development Checks

If test dependencies are installed:

```powershell
python -m unittest discover -s tests
python -m ruff check .
```

GitHub CI installs the smaller `requirements-ci.txt` set so it can lint and run
the smoke tests without downloading the IBM/Qiskit/Jupyter stack.

Optional broader checks:

```powershell
python -m compileall Main Main2 Baselines python_library
```

## Current Research Caveats

- Most quantitative comparisons are simulator studies; the separate five-image
  IBM Heron pilot executes the measurement circuits on a QPU and is reported as
  feasibility evidence, not as a hardware quantum-advantage benchmark.
- `lightning.gpu` is optional and environment-dependent.
- CIFAR runs use small grayscale 32x32 samples to keep the experiments quick.
- The benchmark methods are intentionally simple. They are comparison anchors,
  not state-of-the-art reconstruction systems.
- The IBM Heron pilot contains real hardware reconstruction paths, but its five
  per-image-optimized examples are too small to support a general performance or
  quantum-advantage claim.
