# Dataset-level generalization study

## Reviewer ask

Replace most per-image headline results with a genuine dataset-level experiment:
hundreds-thousands of training images, a fixed model trained across images, real
validation/test splits, multiple seeds, unseen-class or distribution-shift evaluation.

## What already existed (before this tracker)

The paper's existing "held-out" numbers (18.80 dB controls vs 18.12 dB HVK2D, cited in
`latex_outputs/paper_latex/supplementary_study.tex`) come from
`Main2/newHVK/run_newhvk_suite.py:run_real_cifar_holdout` (line ~1004), which:

- Uses a **hand-crafted closed-form feature transform** (`real_newhvk_features`, line
  890 — pairwise products + sine harmonics of classical MPS features), **not** the real
  gradient-trained `QuantumModel`/`Quantum2DGridModel` circuit.
- Fits with **ridge regression** (`ridge_fit_predict`, line 48), not SGD.
- Uses only **6 training images + 4 held-out images per seed** (5 seeds).

No script anywhere in the repo trained a single shared model via SGD across many images
with a real train/held-out split before this effort. Every gradient-trained model in
`Baselines/cifar10_comparisons/*/run_*.py` fits one image at a time.

Cached data: `Baselines/cifar10_comparisons/datasets/images/` already has 1000 CIFAR-10
test-split PNGs (`manifest.json`), far more than needed for the current scope — no new
data acquisition required.

## What this effort adds

`Main2/newHVK/run_dataset_level_generalization.py` (new): trains ONE
`Quantum2DGridModel` + `PatchDecoder2D` (the paper's actual HVK2D architecture, not the
ridge-regression proxy) via full-batch Adam across all patches from N_TRAIN images, then
evaluates with no further training on N_HELDOUT images never seen during training.
Normalization statistics computed from the training split only (no leakage). A
parameter-matched `ClassicalLinearControl` (same decoder, same observable width, single
linear feature map, no quantum circuit) is trained identically on the same split for a
fair head-to-head comparison.

Scope chosen (2026-07-27, confirmed with project owner): 150 train / 50 held-out images,
3 seeds (each reshuffling the split), 20 full-batch epochs. Reduced from an initial
150/50/3-seed/30-epoch plan after a timing probe showed the real per-patch circuit cost
is ~137ms (not ~96ms as first estimated from a different qubit-count timing test), to
keep total compute to roughly 5.5 hours. This is a training-budget constraint, not a
claim of full convergence — report it as such in any write-up.

Still NOT covered by this scope: unseen-class / distribution-shift evaluation (the
reviewer's stretch goal). Current split is random-stratified unseen *images*, same
classes as training, matching the existing (smaller) held-out study's protocol. A true
unseen-class variant would need a class-aware split — flag to project owner as a
possible follow-up once this run's results are in.

## Status

- Script written and smoke-tested (`--n-train 5 --n-heldout 5 --epochs 3`) — runs correctly
  end-to-end, produces held-out PSNR/MSE/SSIM per image and per model.
- Full run (150/50, 3 seeds, 20 epochs) queued to launch once the concurrent
  qubit-energy phase-transition sweep (see `../07_phase_transition_scope/`) finishes, to
  avoid CPU contention between two compute-bound background jobs.
- `--device` defaults to `cpu`: PennyLane's `default.qubit` doesn't run on GPU, so moving
  the many tiny per-patch tensors to CUDA only adds transfer overhead (measured ~2.5x
  slower on this machine — learned the hard way on the phase-transition sweep, see that
  folder's README).

## Output location

`Main2/newHVK/results/dataset_level_generalization/dataset_level_generalization.json`
(per-seed raw results, resumable) and `..._summary.json` (mean±std held-out PSNR across
seeds, for both `hvk2d_quantum` and `classical_linear_control`).

## Next step once the run completes

1. Compare `hvk2d_quantum` vs `classical_linear_control` held-out PSNR mean±std across
   3 seeds — this is the actual number to report, replacing (or standing alongside) the
   6+4-image ridge-regression proxy result.
2. Write a new subsection into `paper_hvk.tex` / `supplementary_study.tex` reporting this
   as the genuine dataset-level generalization result, explicit about scope (150/50/3
   seeds, 20-epoch budget) vs the reviewer's literal "hundreds-thousands" ask.
3. Feed the outcome into `../01_narrative_reframe/` — if the quantum model still doesn't
   beat the classical control at this larger scale, that strengthens (not weakens) the
   "boundary study" reframing the reviewer recommends, since it would no longer be
   attributable to the earlier study's tiny 6-image sample size.
