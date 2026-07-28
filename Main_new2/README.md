# Main_new2: HVK2D with a working Hamiltonian objective

Companion to `Main_new/` (see that folder's README for the full diagnosis) — same fix,
applied to the HVK2D grid-topology model instead of HVK1D's chain model.

This is a scoped rebuild of `Main2/src/` (the core HVK2D model + training loop), not a
copy of the much larger `Main2/newHVK/` research-script collection, which contains dozens
of unrelated experiments (finite-size scaling, ablation suites, symmetry checks, etc.)
that don't touch the Hamiltonian objective.

## Fix

1. **Bounded coupling** (`src/model.py`): `Quantum2DGridModel.j_2d` is now a read-only
   property returning `torch.tanh(j_2d_raw)` of an underlying unconstrained parameter —
   the original `j_2d` fed only the energy loss (never the decoder's observables) and
   could grow without limit under a linear energy loss, exactly the same pathology
   diagnosed in `Main_new/`.
2. **Bounded energy loss** (`src/training.py`): `run_main2` now supports
   `energy_loss_mode` (`linear` / `positive` / `contrastive`, matching `Main_new`'s
   HVK1D implementation), defaulting to `positive` (`torch.mean(energies.square())`).
   The original `Main2/src/training.py` only ever computed the unbounded linear energy
   loss — this mode selection did not exist for HVK2D before this folder.

## Bug fixed along the way

`Main2/src/training.py`'s `run_main2` called
`resolve_device(config.device, requires_quantum=True)`, but the `resolve_device` it
actually imports (`Main/src/training/training.resolve_device`) does not accept a
`requires_quantum` keyword — that call would raise `TypeError` if ever exercised. Fixed
in `Main_new2/src/training.py` by dropping the unsupported kwarg. Worth flagging to the
project owner: this suggests `Main2/src/training.py`'s `run_main2` may never have been
successfully run in its current form (the real per-image HVK2D results in the paper come
from `Baselines/cifar10_comparisons/hvk2d/run_hvk2d_cifar32.py` and
`Main2/newHVK/`-family scripts, not this module).

## Entry point

```powershell
python Main_new2\main.py --steps 200 --energy-loss-mode positive
```

Smoke-tested at `--steps 5` (2026-07-28): runs end-to-end correctly, defaults to
`positive` energy-loss mode.

## Status

Built and smoke-tested. Full multi-seed validation against a Table-X-equivalent HVK2D
protocol not yet run (Table X's Hamiltonian ablation was originally single-image,
single-seed HVK1D on Monalisa — an HVK2D-equivalent comparison protocol needs to be
decided: same Monalisa image at native resolution, or the CIFAR images used elsewhere
for HVK2D). Flag to project owner before running a full comparison sweep here.

## What's unchanged

`src/analysis.py`, `src/outputs.py`, `src/config.py`, `src/dataset.py`, `src/pathing.py`
mirror `Main2/src/` exactly except for import-path updates (`Main2.src.X` ->
`Main_new2.src.X`) and `pathing.py` now points at `Main_new/` instead of `Main/`, so this
folder's HVK2D model shares preprocessing/reconstruction utilities with `Main_new`'s
redesigned HVK1D rather than the original `Main/`.
