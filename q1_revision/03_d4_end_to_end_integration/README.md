# D4 end-to-end integration

## Reviewer ask

Integrate D4 pooling into the complete trainable model and demonstrate: output-level
equivariance, sample efficiency, robustness to rotations/reflections, and comparison with
an identically pooled classical baseline. Currently "the paper itself acknowledges that
D4 pooling is a feature-grid diagnostic rather than part of the end-to-end trained
reconstruction pipeline" — so "with Provable Symmetry" in the title overstates the
system-level result (see `../08_title_and_claims/`).

## What already exists

The D4-pooled observable-grid map (`\Phi_{D_4}(x)=\frac{1}{8}\sum_{g\in D_4}P_g^{-1}\Phi(gx)`,
paper Section on D4 symmetry) is verified as a numerical/feature-grid-level construction
(equivariance error ~1e-17) over cached CIFAR patch transforms — it is NOT currently
wired into the trainable reconstruction pipeline (`QuantumModel`/`Quantum2DGridModel` +
decoder). Find the current verification script via the D4 experiment family:
`Main2/newHVK/run_d4_symmetry_experiment.py`, `run_d4_real_circuit_confirmation.py`.

## Gap

Need to: (1) build a pooled-observable variant of the trainable model (apply the D4
group-average to the observable vector before the decoder, for all 8 square symmetries
of each patch), (2) train it end-to-end on real images, (3) measure whether
reconstruction is actually equivariant to rotated/reflected inputs at the *output* level
(not just the feature level), (4) compare sample efficiency and robustness against an
identically-pooled classical (non-quantum) baseline.

This is a real architecture change (8x more circuit evaluations per patch during
training unless the group-average is computed more cleverly, e.g. only at eval time) —
likely the second-most compute-heavy item in this tracker after
`../02_dataset_level_generalization/`. Not started; needs scoping (which model variant,
how many images/seeds) before launching anything.

## Status

Not started.
