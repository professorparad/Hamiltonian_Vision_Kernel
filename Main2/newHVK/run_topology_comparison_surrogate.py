"""Workstream 1 (surrogate sweep): HVK1D vs HVK2D topology comparison, full
statistical scope (5 seeds x held-out splits x 3 datasets), reusing the
already-verified analytic-surrogate held-out framework from
run_multi_dataset_validation.py / run_newhvk_suite.py.

The existing HVK2D surrogate (`real_newhvk_features`) is a closed-form proxy
built from 6 specific index-pairs of the 18-wide local-observable block,
standing in for the trained HVK2D grid circuit's pairwise ZZ correlators.
This script adds an HONEST HVK1D-equivalent (`real_hvk1d_features`), NOT a
relabeled copy: it iterates the chain topology's actual 5 nearest-neighbor
bonds (vs. the grid's edge list) and adds a second pair-type term reflecting
HVK1D's extra XX/YY measurement channels that the 2D grid circuit does not
have (it measures only Z, X, ZZ). Both surrogates share the same local/
positional input convention so the comparison is apples-to-apples.

This is the fast, CPU-only, full-statistical-power leg. The slower
real-trained-circuit confirmation subset is run_topology_comparison.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")
NEWHVK_DIR = REPO_ROOT / "Main2" / "newHVK"
for p in (NEWHVK_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_extended_validation import aggregate, extract_patch_table
from run_multi_dataset_validation import (
    bootstrap_ci,
    load_medmnist_dataset,
    load_torchvision_dataset,
    normalize_images,
    paired_wilcoxon_pvalue,
    reconstruct_images_from_patches,
    stratified_indices,
)
from run_newhvk_suite import (
    RESULTS,
    image_metric_rows,
    real_local_observables_only,
    real_newhvk_features,
    real_no_entanglement_features,
    real_parameter_matched_classical_features,
    real_quadratic_classical_features,
    real_raw_linear_features,
    ridge_fit_predict,
    standardize_train_test,
    select_same_width,
)

OUT_DIR = RESULTS / "topology_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

# HVK1D chain: 5 nearest-neighbor bonds, reusing the same per-qubit index
# convention already used by real_newhvk_features's grid-edge pairs.
CHAIN_BONDS = [(0, 1), (4, 5), (7, 8), (12, 13), (14, 15)]


def real_hvk1d_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    pos = base[:, 18:26]
    zz_pairs = np.stack([local[:, u] * local[:, v] for u, v in CHAIN_BONDS], axis=1)
    # HVK1D's real circuit additionally measures XX and YY per bond (the
    # HVK2D grid circuit measures only Z, X, ZZ) -- reflect that extra
    # measurement-basis richness with a second, distinct pair-type term
    # rather than only reusing the ZZ-style product.
    xx_pairs = np.stack([np.cos(local[:, u]) * np.cos(local[:, v]) for u, v in CHAIN_BONDS], axis=1)
    harmonics = np.sin(np.pi * zz_pairs)
    return select_same_width(np.concatenate([local, zz_pairs, xx_pairs, harmonics, pos], axis=1), 32)


VARIANTS = [
    ("HVK1D-pair-observable", lambda base, seed: real_hvk1d_features(base)),
    ("HVK2D-pair-observable", lambda base, seed: real_newhvk_features(base)),
    ("no-entanglement", lambda base, seed: real_no_entanglement_features(base)),
    ("local-only", lambda base, seed: real_local_observables_only(base)),
    ("raw-linear", lambda base, seed: real_raw_linear_features(base)),
    ("parameter-matched", real_parameter_matched_classical_features),
    ("quadratic-classical", lambda base, seed: real_quadratic_classical_features(base)),
]


def evaluate_reconstruction(dataset_key, images, labels, source, seeds, train_per_class, test_per_class):
    rows = []
    image_size = int(images.shape[1])
    patch_size = 8 if image_size == 32 else 7
    for seed in seeds:
        train_idx, test_idx = stratified_indices(labels, seed, train_per_class, test_per_class)
        if not train_idx or not test_idx:
            continue
        x_train, y_train, _ = extract_patch_table(images[train_idx], patch_size=patch_size)
        x_test, y_test, patch_image_ids = extract_patch_table(images[test_idx], patch_size=patch_size)
        x_train, x_test = standardize_train_test(x_train, x_test)
        for model, feature_fn in VARIANTS:
            pred = ridge_fit_predict(feature_fn(x_train, seed), y_train, feature_fn(x_test, seed))
            pred_images, target_images = reconstruct_images_from_patches(
                pred, y_test, patch_image_ids, image_size=image_size, patch_size=patch_size
            )
            for image_id in sorted(pred_images):
                metrics = image_metric_rows(pred_images[image_id], target_images[image_id])
                rows.append({
                    "dataset": dataset_key, "source": source, "seed": seed, "model": model,
                    "image_id": image_id, "train_images": len(train_idx), "test_images": len(test_idx),
                    **metrics,
                })
    return rows, aggregate(rows) if rows else []


def evaluate_classification(dataset_key, images, labels, source, seeds, train_per_class, test_per_class):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from run_multi_dataset_validation import image_level_base_features

    rows = []
    image_size = int(images.shape[1])
    patch_size = 8 if image_size == 32 else 7
    base = image_level_base_features(images, patch_size)
    for seed in seeds:
        train_idx, test_idx = stratified_indices(labels, seed, train_per_class, test_per_class)
        if not train_idx or not test_idx:
            continue
        x_train_base, x_test_base = standardize_train_test(base[train_idx], base[test_idx])
        y_train, y_test = labels[train_idx], labels[test_idx]
        for model, feature_fn in VARIANTS:
            x_train = feature_fn(x_train_base, seed)
            x_test = feature_fn(x_test_base, seed)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            rows.append({
                "dataset": dataset_key, "source": source, "seed": seed, "model": model,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
            })
    if not rows:
        return rows, []
    summary = []
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        summary.append({
            "model": model, "n": len(mrows),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in mrows])),
            "std_accuracy": float(np.std([r["accuracy"] for r in mrows])),
        })
    return rows, sorted(summary, key=lambda r: -r["mean_accuracy"])


def paired_stats(rows, dataset_names):
    stats = []
    controls = ["HVK2D-pair-observable", "local-only", "raw-linear", "no-entanglement", "quadratic-classical", "parameter-matched"]
    for dataset in dataset_names:
        drows = [r for r in rows if r["dataset"] == dataset]
        hvk1d = {(r["seed"], r["image_id"]): r for r in drows if r["model"] == "HVK1D-pair-observable"}
        for control in controls:
            ctrl = {(r["seed"], r["image_id"]): r for r in drows if r["model"] == control}
            keys = sorted(set(hvk1d) & set(ctrl))
            if not keys:
                continue
            diff = np.asarray([hvk1d[k]["psnr"] - ctrl[k]["psnr"] for k in keys])
            low, high = bootstrap_ci(diff, seed=200_000 + len(stats))
            stats.append({
                "dataset": dataset, "comparison": f"HVK1D-pair-observable minus {control}",
                "n_pairs": len(keys), "mean_psnr_difference_db": float(diff.mean()),
                "bootstrap95_low_db": low, "bootstrap95_high_db": high,
                "wilcoxon_p_psnr": paired_wilcoxon_pvalue(diff),
            })
    return stats


def load_cifar10_labeled(limit: int):
    cifar_dir = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "datasets" / "images"
    paths = sorted(cifar_dir.glob("*.png"))[:limit]
    import matplotlib.pyplot as plt

    images, labels = [], []
    for path in paths:
        image = plt.imread(path)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=-1)
        images.append(image)
        parts = path.name.split("_")
        labels.append(parts[1] if len(parts) > 1 else "unknown")
    _, numeric = np.unique(np.asarray(labels), return_inverse=True)
    return normalize_images(np.asarray(images)), numeric, str(cifar_dir)


def main():
    datasets = []
    datasets.append(("cifar10-native32", *load_cifar10_labeled(limit=200), 1, 1))
    datasets.append(("fashion-mnist", *load_torchvision_dataset("fashion-mnist", download=False, limit=800), 20, 10))
    datasets.append(("pathmnist", *load_medmnist_dataset("pathmnist", download=False, limit=800), 20, 10))

    all_recon_rows, all_cls_rows = [], []
    manifest = {}
    for dataset_key, images, labels, source, train_pc, test_pc in datasets:
        print(f"\n=== {dataset_key} (source={source}, n_images={images.shape[0]}, n_classes={len(set(labels.tolist()))}) ===")
        recon_rows, recon_summary = evaluate_reconstruction(dataset_key, images, labels, source, SEEDS, train_pc, test_pc)
        cls_rows, cls_summary = evaluate_classification(dataset_key, images, labels, source, SEEDS, train_pc, test_pc)
        print("  reconstruction summary:", json.dumps(recon_summary, indent=2)[:800])
        print("  classification summary:", json.dumps(cls_summary, indent=2)[:800])
        all_recon_rows.extend(recon_rows)
        all_cls_rows.extend(cls_rows)
        manifest[dataset_key] = {"source": source, "n_images": int(images.shape[0]), "n_recon_rows": len(recon_rows), "n_cls_rows": len(cls_rows)}
        (OUT_DIR / f"{dataset_key}_reconstruction.json").write_text(json.dumps(recon_rows, indent=2))
        (OUT_DIR / f"{dataset_key}_reconstruction_summary.json").write_text(json.dumps(recon_summary, indent=2))
        (OUT_DIR / f"{dataset_key}_classification.json").write_text(json.dumps(cls_rows, indent=2))
        (OUT_DIR / f"{dataset_key}_classification_summary.json").write_text(json.dumps(cls_summary, indent=2))

    stats = paired_stats(all_recon_rows, [d[0] for d in datasets])
    (OUT_DIR / "surrogate_paired_stats.json").write_text(json.dumps(stats, indent=2))
    (OUT_DIR / "surrogate_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nDone. Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
