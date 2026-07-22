from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_extended_validation import (
    OUT,
    aggregate,
    binary_metrics,
    extract_patch_table,
    logistic_fit_predict,
    run_wisconsin_breast_cancer,
    write_dict_csv,
)
from run_newhvk_suite import (
    ROOT,
    WORKSPACE,
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
)


OUT_MULTI = RESULTS / "multi_dataset_validation"


def normalize_images(images: np.ndarray, limit: int | None = None) -> np.ndarray:
    arr = np.asarray(images)
    if limit is not None:
        arr = arr[:limit]
    arr = arr.astype(np.float64)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] in {3, 4}:
            arr = arr[..., :3].mean(axis=-1)
    if arr.max(initial=0.0) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def load_npz_dataset(path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    data = np.load(path)
    if {"images", "labels"}.issubset(data.files):
        images = data["images"]
        labels = data["labels"]
    elif {"x_train", "y_train"}.issubset(data.files):
        images = data["x_train"]
        labels = data["y_train"]
    elif {"train_images", "train_labels"}.issubset(data.files):
        images = data["train_images"]
        labels = data["train_labels"]
    else:
        raise ValueError(f"Unsupported NPZ dataset keys in {path}: {data.files}")
    return normalize_images(images, limit), np.asarray(labels).reshape(-1).astype(int)[:limit], f"local-npz:{path}"


def load_torchvision_dataset(name: str, download: bool, limit: int | None) -> tuple[np.ndarray, np.ndarray, str]:
    from torchvision import datasets

    root = WORKSPACE / "datasets"
    if name == "mnist":
        dataset = datasets.MNIST(root=str(root), train=True, download=download)
    elif name == "fashion-mnist":
        dataset = datasets.FashionMNIST(root=str(root), train=True, download=download)
    else:
        raise ValueError(name)
    images = dataset.data.numpy() if hasattr(dataset.data, "numpy") else np.asarray(dataset.data)
    labels = dataset.targets.numpy() if hasattr(dataset.targets, "numpy") else np.asarray(dataset.targets)
    return normalize_images(images, limit), labels[:limit].astype(int), f"torchvision-{name}:{root}"


def load_medmnist_dataset(name: str, download: bool, limit: int | None) -> tuple[np.ndarray, np.ndarray, str]:
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    dataset_class = getattr(medmnist, info["python_class"])
    root = WORKSPACE / "datasets"
    dataset = dataset_class(split="train", root=str(root), download=download, as_rgb=False, size=28)
    images = np.asarray(dataset.imgs)
    labels = np.asarray(dataset.labels).reshape(-1)
    return normalize_images(images, limit), labels[:limit].astype(int), f"medmnist-{name}:{root}"


def stratified_indices(labels: np.ndarray, seed: int, train_per_class: int, test_per_class: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(set(labels.tolist())):
        candidates = np.where(labels == label)[0]
        if len(candidates) < train_per_class + test_per_class:
            continue
        order = rng.permutation(candidates)
        train_idx.extend(order[:train_per_class])
        test_idx.extend(order[train_per_class : train_per_class + test_per_class])
    return train_idx, test_idx


def reconstruct_images_from_patches(
    pred: np.ndarray,
    target: np.ndarray,
    patch_image_ids: list[int],
    image_size: int,
    patch_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    pred_images: dict[int, np.ndarray] = {}
    target_images: dict[int, np.ndarray] = {}
    patches_per_row = image_size // patch_size
    for patch_index, image_id in enumerate(patch_image_ids):
        local_index = sum(1 for prior_id in patch_image_ids[:patch_index] if prior_id == image_id)
        row_block = local_index // patches_per_row
        col_block = local_index % patches_per_row
        row = row_block * patch_size
        col = col_block * patch_size
        pred_images.setdefault(image_id, np.zeros((image_size, image_size), dtype=np.float64))
        target_images.setdefault(image_id, np.zeros((image_size, image_size), dtype=np.float64))
        pred_images[image_id][row : row + patch_size, col : col + patch_size] = np.clip(
            pred[patch_index].reshape(patch_size, patch_size), 0.0, 1.0
        )
        target_images[image_id][row : row + patch_size, col : col + patch_size] = target[patch_index].reshape(
            patch_size, patch_size
        )
    return pred_images, target_images


def evaluate_image_dataset(
    dataset_key: str,
    images: np.ndarray,
    labels: np.ndarray,
    source: str,
    seeds: list[int],
    train_per_class: int,
    test_per_class: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    variants = [
        ("HVK2D-pair-observable", lambda base, seed: real_newhvk_features(base)),
        ("no-entanglement", lambda base, seed: real_no_entanglement_features(base)),
        ("local-only", lambda base, seed: real_local_observables_only(base)),
        ("raw-linear", lambda base, seed: real_raw_linear_features(base)),
        ("parameter-matched", real_parameter_matched_classical_features),
        ("quadratic-classical", lambda base, seed: real_quadratic_classical_features(base)),
    ]
    image_size = int(images.shape[1])
    patch_size = 8 if image_size == 32 else 7
    for seed in seeds:
        train_idx, test_idx = stratified_indices(labels, seed, train_per_class, test_per_class)
        if not train_idx or not test_idx:
            continue
        train_images = images[train_idx]
        test_images = images[test_idx]
        x_train, y_train, _ = extract_patch_table(train_images, patch_size=patch_size)
        x_test, y_test, patch_image_ids = extract_patch_table(test_images, patch_size=patch_size)
        x_train, x_test = standardize_train_test(x_train, x_test)
        for model, feature_fn in variants:
            pred = ridge_fit_predict(feature_fn(x_train, seed), y_train, feature_fn(x_test, seed))
            pred_images, target_images = reconstruct_images_from_patches(
                pred, y_test, patch_image_ids, image_size=image_size, patch_size=patch_size
            )
            for image_id in sorted(pred_images):
                metrics = image_metric_rows(pred_images[image_id], target_images[image_id])
                rows.append(
                    {
                        "dataset": dataset_key,
                        "source": source,
                        "seed": seed,
                        "model": model,
                        "image_id": image_id,
                        "train_images": len(train_idx),
                        "test_images": len(test_idx),
                        **metrics,
                    }
                )
    if not rows:
        return rows, []
    return rows, aggregate(rows)


def bootstrap_ci(values: np.ndarray, seed: int = 1234, n_bootstrap: int = 3000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        means[index] = float(rng.choice(values, size=values.size, replace=True).mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_wilcoxon_pvalue(values: np.ndarray) -> float:
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        signs = int(np.sum(values > 0.0))
        nonzero = int(np.sum(np.abs(values) > 1e-12))
        if nonzero == 0:
            return 1.0
        tail = min(signs, nonzero - signs)
        probability = sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2**nonzero)
        return float(min(1.0, 2.0 * probability))


def write_multi_dataset_paired_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    controls = ["local-only", "raw-linear", "no-entanglement", "quadratic-classical", "parameter-matched"]
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        hvk = {
            (int(row["seed"]), int(row["image_id"])): row
            for row in dataset_rows
            if row["model"] == "HVK2D-pair-observable"
        }
        for control in controls:
            ctrl = {
                (int(row["seed"]), int(row["image_id"])): row
                for row in dataset_rows
                if row["model"] == control
            }
            keys = sorted(set(hvk).intersection(ctrl))
            if not keys:
                continue
            image_psnr_diff = {
                key: float(hvk[key]["psnr"]) - float(ctrl[key]["psnr"])
                for key in keys
            }
            image_mse_diff = {
                key: float(hvk[key]["mse"]) - float(ctrl[key]["mse"])
                for key in keys
            }
            # Held-out images produced by one fitted split share the same
            # readout. Aggregate within seed before inferential statistics.
            seeds = sorted({seed for seed, _ in keys})
            psnr_diff = np.asarray([
                np.mean([value for (row_seed, _), value in image_psnr_diff.items() if row_seed == seed])
                for seed in seeds
            ])
            mse_diff = np.asarray([
                np.mean([value for (row_seed, _), value in image_mse_diff.items() if row_seed == seed])
                for seed in seeds
            ])
            low, high = bootstrap_ci(psnr_diff, seed=150_000 + len(stats))
            stats.append(
                {
                    "dataset": dataset,
                    "comparison": f"HVK2D-pair-observable minus {control}",
                    "n_seeds": len(seeds),
                    "n_image_seed_pairs": len(keys),
                    "inference_unit": "seed mean over held-out images",
                    "mean_psnr_difference_db": float(psnr_diff.mean()),
                    "bootstrap95_low_db": low,
                    "bootstrap95_high_db": high,
                    "wilcoxon_p_psnr": paired_wilcoxon_pvalue(psnr_diff),
                    "mean_mse_difference": float(mse_diff.mean()),
                    "interpretation": "positive PSNR difference favors HVK2D; negative favors the control",
                }
            )
    write_dict_csv(OUT_MULTI / "all_image_datasets_paired_stats.csv", stats)
    (OUT_MULTI / "all_image_datasets_paired_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def image_level_base_features(images: np.ndarray, patch_size: int) -> np.ndarray:
    image_features: list[np.ndarray] = []
    for image in images:
        x, _, _ = extract_patch_table(np.asarray([image]), patch_size=patch_size)
        image_features.append(np.concatenate([x.mean(axis=0), x.std(axis=0)], axis=0))
    return np.asarray(image_features, dtype=np.float64)


def classify_dataset(
    dataset_key: str,
    images: np.ndarray,
    labels: np.ndarray,
    source: str,
    seeds: list[int],
    train_per_class: int,
    test_per_class: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
    except Exception as exc:
        rows = [
            {
                "dataset": dataset_key,
                "source": source,
                "seed": 0,
                "model": "sklearn-unavailable",
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "notes": str(exc),
            }
        ]
        return rows, aggregate(rows)
    rows: list[dict[str, object]] = []
    image_size = int(images.shape[1])
    patch_size = 8 if image_size == 32 else 7
    variants = [
        ("HVK2D-pair-observable", lambda base, seed: real_newhvk_features(base)),
        ("no-entanglement", lambda base, seed: real_no_entanglement_features(base)),
        ("local-only", lambda base, seed: real_local_observables_only(base)),
        ("raw-linear", lambda base, seed: real_raw_linear_features(base)),
        ("quadratic-classical", lambda base, seed: real_quadratic_classical_features(base)),
        ("parameter-matched", real_parameter_matched_classical_features),
    ]
    base = image_level_base_features(images, patch_size)
    for seed in seeds:
        train_idx, test_idx = stratified_indices(labels, seed, train_per_class, test_per_class)
        if not train_idx or not test_idx:
            continue
        x_train_base, x_test_base = standardize_train_test(base[train_idx], base[test_idx])
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        for model, feature_fn in variants:
            x_train = feature_fn(x_train_base, seed)
            x_test = feature_fn(x_test_base, seed)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            rows.append(
                {
                    "dataset": dataset_key,
                    "source": source,
                    "seed": seed,
                    "model": model,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro")),
                }
            )
    if not rows:
        return rows, []
    summary: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        summary.append(
            {
                "model": model,
                "n": len(model_rows),
                "mean_accuracy": float(np.mean([float(row["accuracy"]) for row in model_rows])),
                "std_accuracy": float(np.std([float(row["accuracy"]) for row in model_rows], ddof=0)),
                "mean_macro_f1": float(np.mean([float(row["macro_f1"]) for row in model_rows])),
                "std_macro_f1": float(np.std([float(row["macro_f1"]) for row in model_rows], ddof=0)),
            }
        )
    return rows, sorted(summary, key=lambda row: -float(row["mean_accuracy"]))


def run_all_image_datasets(download: bool, limit: int, seeds: list[int]) -> dict[str, object]:
    OUT_MULTI.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"download_requested": download, "datasets": {}}
    datasets_to_run: list[tuple[str, np.ndarray, np.ndarray, str]] = []

    cifar_script = ROOT / "Baselines" / "cifar10_comparisons" / "download_cifar32.py"
    if download:
        subprocess.run(
            [
                sys.executable,
                str(cifar_script),
                "--output-dir",
                str(ROOT / "Baselines" / "cifar10_comparisons" / "datasets"),
                "--count",
                str(max(limit, 100)),
                "--split",
                "test",
            ],
            check=True,
        )

    cifar_paths = sorted((ROOT / "Baselines" / "cifar10_comparisons" / "datasets" / "images").glob("*.png"))
    if cifar_paths:
        import matplotlib.pyplot as plt

        images = []
        labels = []
        for path in cifar_paths[:limit]:
            image = plt.imread(path)
            if image.ndim == 3:
                image = image[..., :3].mean(axis=-1)
            images.append(image)
            parts = path.name.split("_")
            labels.append(parts[1] if len(parts) > 1 else "unknown")
        _, numeric_labels = np.unique(np.asarray(labels), return_inverse=True)
        datasets_to_run.append(("cifar10-native32", normalize_images(np.asarray(images)), numeric_labels, str(cifar_paths[0].parent)))

    for name in ["mnist", "fashion-mnist"]:
        try:
            datasets_to_run.append((name, *load_torchvision_dataset(name, download=download, limit=limit)))
        except Exception as exc:
            manifest["datasets"][name] = {"status": "unavailable", "error": str(exc)}

    for name in ["pathmnist", "bloodmnist", "pneumoniamnist"]:
        try:
            local_path = WORKSPACE / "datasets" / f"{name}.npz"
            if local_path.exists():
                datasets_to_run.append((name, *load_npz_dataset(local_path, limit=limit)))
            else:
                datasets_to_run.append((name, *load_medmnist_dataset(name, download=download, limit=limit)))
        except Exception as exc:
            manifest["datasets"][name] = {"status": "unavailable", "error": str(exc)}

    combined_rows: list[dict[str, object]] = []
    combined_summary: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    classification_summary: list[dict[str, object]] = []
    for dataset_key, images, labels, source in datasets_to_run:
        train_per_class = 20 if dataset_key != "cifar10-native32" else 1
        test_per_class = 10 if dataset_key != "cifar10-native32" else 1
        rows, summary = evaluate_image_dataset(
            dataset_key,
            images,
            labels,
            source,
            seeds=seeds,
            train_per_class=train_per_class,
            test_per_class=test_per_class,
        )
        dataset_dir = OUT_MULTI / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_dict_csv(dataset_dir / f"{dataset_key}.csv", rows)
        write_dict_csv(dataset_dir / f"{dataset_key}_summary.csv", summary)
        (dataset_dir / f"{dataset_key}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (dataset_dir / f"{dataset_key}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        combined_rows.extend(rows)
        for row in summary:
            combined_summary.append({"dataset": dataset_key, **row})
        cls_rows, cls_summary = classify_dataset(
            dataset_key,
            images,
            labels,
            source,
            seeds=seeds,
            train_per_class=train_per_class,
            test_per_class=test_per_class,
        )
        cls_dir = OUT_MULTI / dataset_key
        write_dict_csv(cls_dir / f"{dataset_key}_classification.csv", cls_rows)
        write_dict_csv(cls_dir / f"{dataset_key}_classification_summary.csv", cls_summary)
        (cls_dir / f"{dataset_key}_classification.json").write_text(json.dumps(cls_rows, indent=2), encoding="utf-8")
        (cls_dir / f"{dataset_key}_classification_summary.json").write_text(json.dumps(cls_summary, indent=2), encoding="utf-8")
        classification_rows.extend(cls_rows)
        for row in cls_summary:
            classification_summary.append({"dataset": dataset_key, **row})
        manifest["datasets"][dataset_key] = {
            "status": "tested" if rows else "skipped-not-enough-stratified-samples",
            "source": source,
            "n_images_loaded": int(images.shape[0]),
            "image_shape": list(images.shape[1:]),
            "n_classes_loaded": int(len(set(labels.tolist()))),
        }
    write_dict_csv(OUT_MULTI / "all_image_datasets.csv", combined_rows)
    write_dict_csv(OUT_MULTI / "all_image_datasets_summary.csv", combined_summary)
    write_multi_dataset_paired_stats(combined_rows)
    write_dict_csv(OUT_MULTI / "all_image_classification.csv", classification_rows)
    write_dict_csv(OUT_MULTI / "all_image_classification_summary.csv", classification_summary)
    return manifest


def run_wisconsin() -> list[dict[str, object]]:
    rows, summary = run_wisconsin_breast_cancer()
    dataset_dir = OUT_MULTI / "wisconsin_breast_cancer"
    write_dict_csv(dataset_dir / "wisconsin_breast_cancer.csv", rows)
    write_dict_csv(dataset_dir / "wisconsin_breast_cancer_summary.csv", summary)
    (dataset_dir / "wisconsin_breast_cancer.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (dataset_dir / "wisconsin_breast_cancer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HVK controls across CIFAR, MNIST, Fashion-MNIST, MedMNIST, and Wisconsin.")
    parser.add_argument("--download", action="store_true", help="Download missing public datasets into main2/newHVK/datasets.")
    parser.add_argument("--limit", type=int, default=400, help="Maximum images loaded per dataset before stratified sampling.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Subset seeds.")
    parser.add_argument("--skip-wisconsin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_MULTI.mkdir(parents=True, exist_ok=True)
    manifest = run_all_image_datasets(download=args.download, limit=args.limit, seeds=args.seeds)
    if not args.skip_wisconsin:
        manifest["wisconsin_breast_cancer"] = {
            "status": "tested",
            "summary": run_wisconsin(),
        }
    (OUT_MULTI / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = """# Multi-dataset validation

Generated by `main2/newHVK/run_multi_dataset_validation.py`.

This runner evaluates the same fixed-width HVK/control feature maps on every
available requested dataset: CIFAR-10 native 32x32 cache, MNIST, Fashion-MNIST,
MedMNIST-style datasets such as PathMNIST/BloodMNIST/PneumoniaMNIST, and the
Wisconsin Breast Cancer tabular benchmark.

Claim boundary: these are lightweight reviewer diagnostics. They enrich the
evidence base, but do not by themselves prove quantum advantage.
"""
    (OUT_MULTI / "README.md").write_text(readme, encoding="utf-8")
    print(f"Multi-dataset validation complete: {OUT_MULTI}")


if __name__ == "__main__":
    main()
