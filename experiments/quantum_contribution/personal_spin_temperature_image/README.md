# Personal quantum spin-temperature image experiment

This folder is intentionally private to this working copy. The repository's
root `.gitignore` excludes `experiments/quantum_contribution/`, including all
code and generated results here.

## Scientific question

Can an image be represented as a two-dimensional spin system and reconstructed
from a thermal quantum state?

For each small image patch, the experiment constructs the transverse-field
Ising Hamiltonian

```text
H = -J sum_<i,j> Z_i Z_j - sum_i h_i Z_i - Gamma sum_i X_i
```

where the local field of an observed grayscale pixel is
`h_i = field_strength * (2*x_i - 1)`. Missing pixels have `h_i = 0`.
At each requested temperature it forms the exact Gibbs state

```text
rho(T) = exp(-H/T) / Tr(exp(-H/T))
```

and reconstructs pixels from measured quantum magnetizations:

```text
x_hat_i = (1 + Tr[rho(T) Z_i]) / 2.
```

It also evaluates a spatial local-energy density, producing the proposed
quantum heat map.

## Important interpretation

This is a **quantum-model simulation**, not execution on quantum hardware.
The Hamiltonian, Gibbs state, and measured observables are quantum mechanical,
but a classical computer performs the exact matrix diagonalization and writes
the output images.

Temperature or local energy alone is not inverted to obtain the image.
Reconstruction uses the site-resolved magnetizations `<Z_i>`. The temperature
controls how strongly the encoded local fields and nearest-neighbour
correlations survive thermal mixing.

Two tasks are saved:

- `encoded`: every pixel supplies a local field. This tests thermal
  encoding/decoding.
- `completion`: only a deterministic subset supplies local fields. This tests
  whether nearest-neighbour quantum correlations infer withheld pixels.

## Run

From the repository root:

```powershell
.\.venv311\Scripts\python.exe experiments\quantum_contribution\personal_spin_temperature_image\run_experiment.py
```

Outputs are written to `results/`, which is also ignored by Git.

## Repeated-mask ablation

The follow-up sweep compares the full transverse-field model against
`Gamma=0` classical Ising, `J=0` local-only controls, and simple classical
image-completion baselines:

```powershell
.\.venv311\Scripts\python.exe experiments\quantum_contribution\personal_spin_temperature_image\run_ablation_sweep.py
```

Its statistical tables, plot, and report are saved under
`results/ablation_sweep/`.

## Overlapping-patch follow-up

This removes hard non-overlapping tile boundaries by evaluating every
stride-one patch and averaging all site measurements that cover each pixel:

```powershell
.\.venv311\Scripts\python.exe experiments\quantum_contribution\personal_spin_temperature_image\run_overlapping_patches.py
```

This creates effective cross-boundary context, while remaining explicit that
the patches are separate six-qubit Gibbs systems rather than one globally
coupled 576-qubit state.

## Frozen held-out validation

The final validation freezes the exploratory optimum and evaluates 30 new
masks without parameter tuning:

```powershell
.\.venv311\Scripts\python.exe experiments\quantum_contribution\personal_spin_temperature_image\run_preregistered_validation.py
```

It writes the hypothesis and decision rule to `preregistration.json` before
evaluating the first held-out mask and checkpoints every completed mask.

## Collector–modulator prototype

The two-stage thermodynamic qubit architecture is specified in
`COLLECTOR_MODULATOR_DESIGN.md` and run with:

```powershell
.\.venv311\Scripts\python.exe experiments\quantum_contribution\personal_spin_temperature_image\run_collector_modulator.py
```

The collector measures two-temperature local spin response. Those measurements
modulate missing-pixel fields, edge couplings, and site-dependent transverse
fields in a second Gibbs Hamiltonian. Because measured observables program the
second stage, this prototype is a hybrid quantum-control protocol even though
both physical stages are quantum models.

## Heat-current TQ-CMR v3

The finite-reservoir version uses the thermodynamic-neuron heat-current ODE:

```powershell
.\.venv311\Scripts\python.exe experiments\quantum_contribution\personal_spin_temperature_image\run_heat_current_tqcmr_v3.py
```

It compares quantum and classical collectors, the unevolved one-pass
collector, and the spatial modulator reservoir while recording final current
residuals to verify thermodynamic convergence.
