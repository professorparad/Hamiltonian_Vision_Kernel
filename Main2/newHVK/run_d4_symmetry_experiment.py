"""Workstream 3 (surrogate leg): end-to-end D4 symmetry experiment.

Extends the existing run_d4_equivariance() (run_extended_validation.py:502,
single-pass, unseeded, 5-image, feature-equivariance-only) with:
  - a held-out train/test protocol (5 seeds, disjoint image splits, reusing
    the same random-permutation pattern already used elsewhere in
    run_extended_validation.py)
  - a genuine classical-equivariant baseline: the SAME Reynolds-averaging
    trick already used for D4-pooled-HVK2D, applied to the classical
    "local-raw" feature map instead of the HVK2D-surrogate one, so the
    comparison is architecture-matched (pool vs. no-pool), not "quantum
    surrogate vs nothing"
  - an ordinary-augmentation baseline: the non-pooled HVK2D-surrogate
    feature map, but with a ridge readout trained on D4-augmented training
    data (each train image's 8 D4 transforms all included as training rows)
    instead of architectural pooling
  - output consistency, not just feature equivariance: a ridge-regression
    readout (features -> patch pixels) is fit per mode, then for each
    held-out image we apply all 8 D4 transforms, predict a reconstruction
    for each, inverse-transform predictions back to canonical orientation,
    and measure (a) pairwise reconstruction MSE across the 8 predictions
    (consistency) and (b) PSNR against ground truth (accuracy).

This is the surrogate (fast, CPU-only) leg; real-circuit confirmation via
exact Qiskit statevector replay on D4-transformed patches is a separate,
smaller follow-up once the GPU is free.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")
NEWHVK_DIR = REPO_ROOT / "Main2" / "newHVK"
for p in (NEWHVK_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_extended_validation import (
    CIFAR_IMAGES,
    d4_transforms,
    grid_feature_map_base,
    inverse_transform_grid,
    load_cifar_gray,
    psnr_from_mse,
)
from run_newhvk_suite import RESULTS, ridge_fit_predict, standardize_train_test

OUT_DIR = RESULTS / "d4_symmetry_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN, N_TEST = 20, 10


def d4_pooled_feature_map(image: np.ndarray, base_mode: str) -> np.ndarray:
    """Reynolds-average grid_feature_map_base(base_mode) over the 8 D4
    transforms -- the exact same construction already used for
    D4-pooled-HVK2D (run_extended_validation.py:488), generalized to any
    base feature map so it can be applied to a classical feature map too."""
    aligned = [
        inverse_transform_grid(grid_feature_map_base(transformed, base_mode), transform)
        for transform, transformed in d4_transforms(image).items()
    ]
    return np.mean(aligned, axis=0)


def feature_map(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "D4-pooled-HVK2D":
        return d4_pooled_feature_map(image, "HVK2D-positional")
    if mode == "D4-pooled-classical":
        return d4_pooled_feature_map(image, "local-raw")
    if mode == "HVK2D-positional" or mode == "HVK2D-augmented":
        return grid_feature_map_base(image, "HVK2D-positional")
    if mode == "local-raw":
        return grid_feature_map_base(image, "local-raw")
    raise ValueError(mode)


MODES = ["D4-pooled-HVK2D", "D4-pooled-classical", "HVK2D-positional", "HVK2D-augmented", "local-raw"]


def equivariance_error(image: np.ndarray, mode: str) -> float:
    base = feature_map(image, mode)
    denom = float(np.linalg.norm(base) + 1e-8)
    errs = []
    for name, transformed_image in d4_transforms(image).items():
        if name == "identity":
            continue
        transformed = inverse_transform_grid(feature_map(transformed_image, mode), name)
        errs.append(float(np.linalg.norm(base - transformed) / denom))
    return float(np.mean(errs))


def patch_rows(image: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the 4x4 grid feature map + raw 8x8 patch targets to 16 rows."""
    grid = feature_map(image, mode)
    feat_dim = grid.shape[-1]
    x = grid.reshape(16, feat_dim)
    y = []
    for row in range(0, 32, 8):
        for col in range(0, 32, 8):
            y.append(image[row : row + 8, col : col + 8].reshape(-1))
    return x, np.asarray(y)


def build_training_set(images: list[np.ndarray], mode: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for image in images:
        if mode == "HVK2D-augmented":
            # ordinary augmentation: every D4 transform of the training image
            # becomes its own training row, using the NON-pooled feature map
            for _, transformed_image in d4_transforms(image).items():
                x, y = patch_rows(transformed_image, "HVK2D-positional")
                xs.append(x)
                ys.append(y)
        else:
            x, y = patch_rows(image, mode)
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def run_mode(mode: str, train_images: list[np.ndarray], test_images: list[np.ndarray]) -> dict:
    x_train, y_train = build_training_set(train_images, mode)
    x_train_std, x_train_mean_std = standardize_train_test(x_train, x_train)
    # recover mean/std used by standardize_train_test for applying to test-time features
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8

    equiv_errors = [equivariance_error(img, mode) for img in test_images]

    consistency_mses, accuracy_psnrs = [], []
    for image in test_images:
        canon_preds = []
        gt_patches = None
        for name, transformed_image in d4_transforms(image).items():
            eval_mode = "HVK2D-positional" if mode == "HVK2D-augmented" else mode
            x_test, y_test = patch_rows(transformed_image, eval_mode)
            x_test_std = (x_test - mean) / std
            pred = ridge_fit_predict(x_train_std, y_train, x_test_std)
            pred_grid = pred.reshape(4, 4, 64)
            pred_canon = inverse_transform_grid(pred_grid, name).reshape(16, 64)
            canon_preds.append(pred_canon)
            if name == "identity":
                gt_patches = y_test
        canon_preds = np.stack(canon_preds, axis=0)  # (8, 16, 64)
        mean_pred = canon_preds.mean(axis=0)
        pairwise_mse = float(np.mean([(p - mean_pred) ** 2 for p in canon_preds]))
        consistency_mses.append(pairwise_mse)
        acc_mse = float(np.mean((mean_pred - gt_patches) ** 2))
        accuracy_psnrs.append(psnr_from_mse(acc_mse))

    return {
        "mode": mode,
        "mean_equivariance_error": float(np.mean(equiv_errors)),
        "std_equivariance_error": float(np.std(equiv_errors)),
        "mean_output_consistency_mse": float(np.mean(consistency_mses)),
        "std_output_consistency_mse": float(np.std(consistency_mses)),
        "mean_accuracy_psnr": float(np.mean(accuracy_psnrs)),
        "std_accuracy_psnr": float(np.std(accuracy_psnrs)),
        "n_test_images": len(test_images),
    }


def main(smoke_test: bool = False):
    paths = sorted(CIFAR_IMAGES.glob("*.png"))
    seeds = [0] if smoke_test else SEEDS
    n_train, n_test = (3, 2) if smoke_test else (N_TRAIN, N_TEST)

    all_rows = []
    for seed in seeds:
        print(f"\n=== seed={seed} ===")
        order = np.random.default_rng(seed).permutation(len(paths))
        train_paths = [paths[i] for i in order[:n_train]]
        test_paths = [paths[i] for i in order[n_train : n_train + n_test]]
        train_images = [load_cifar_gray(p) for p in train_paths]
        test_images = [load_cifar_gray(p) for p in test_paths]
        for mode in MODES:
            row = run_mode(mode, train_images, test_images)
            row["seed"] = seed
            print(json.dumps(row))
            all_rows.append(row)
            (OUT_DIR / "d4_symmetry_experiment.json").write_text(json.dumps(all_rows, indent=2))

    summary = []
    for mode in MODES:
        mode_rows = [r for r in all_rows if r["mode"] == mode]
        if not mode_rows:
            continue
        summary.append({
            "mode": mode,
            "n_seeds": len(mode_rows),
            "mean_equivariance_error": float(np.mean([r["mean_equivariance_error"] for r in mode_rows])),
            "std_equivariance_error": float(np.std([r["mean_equivariance_error"] for r in mode_rows])),
            "mean_output_consistency_mse": float(np.mean([r["mean_output_consistency_mse"] for r in mode_rows])),
            "std_output_consistency_mse": float(np.std([r["mean_output_consistency_mse"] for r in mode_rows])),
            "mean_accuracy_psnr": float(np.mean([r["mean_accuracy_psnr"] for r in mode_rows])),
            "std_accuracy_psnr": float(np.std([r["mean_accuracy_psnr"] for r in mode_rows])),
        })
    (OUT_DIR / "d4_symmetry_experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print("\nDone. Saved to", OUT_DIR)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
