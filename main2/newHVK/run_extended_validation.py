from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_newhvk_suite import (
    CIFAR_IMAGES,
    RESULTS,
    ROOT,
    WORKSPACE,
    image_metric_rows,
    load_cifar_gray,
    psnr_from_mse,
    real_local_observables_only,
    real_newhvk_features,
    real_no_entanglement_features,
    real_parameter_matched_classical_features,
    real_quadratic_classical_features,
    real_raw_linear_features,
    ridge_fit_predict,
    r2_score,
    select_same_width,
    standardize_train_test,
)


OUT = RESULTS / "extended_validation"

DATASET_ROOT_CANDIDATES = [
    WORKSPACE / "datasets",
    ROOT / "data",
    ROOT / "datasets",
    Path.home() / ".cache" / "torch",
    Path.home() / ".cache" / "torchvision",
    Path.home() / ".keras" / "datasets",
]

MEDMNIST_ROOT_CANDIDATES = [
    Path(os.environ["MEDMNIST_ROOT"]) if os.environ.get("MEDMNIST_ROOT") else None,
    WORKSPACE / "datasets",
    WORKSPACE / "medmnist",
    WORKSPACE / "medminist",
    ROOT / "medmnist",
    ROOT / "medminist",
    ROOT / "data",
    ROOT / "datasets",
    Path.home() / ".medmnist",
    Path.home() / ".cache" / "medmnist",
]


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]], group_key: str = "model") -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for group in sorted({str(row[group_key]) for row in rows}):
        group_rows = [row for row in rows if str(row[group_key]) == group]
        item: dict[str, object] = {group_key: group, "n": len(group_rows)}
        for metric in [
            "mse",
            "psnr",
            "ssim",
            "r2",
            "equivariance_error",
            "accuracy",
            "auc",
            "f1",
            "recall",
            "specificity",
            "brier",
        ]:
            values = [float(row[metric]) for row in group_rows if metric in row]
            if values:
                item[f"mean_{metric}"] = float(np.mean(values))
                item[f"std_{metric}"] = float(np.std(values, ddof=0))
        summary.append(item)
    if "mean_mse" in summary[0]:
        return sorted(summary, key=lambda row: float(row["mean_mse"]))
    if "mean_brier" in summary[0]:
        return sorted(summary, key=lambda row: float(row["mean_brier"]))
    if "mean_equivariance_error" in summary[0]:
        return sorted(summary, key=lambda row: float(row["mean_equivariance_error"]))
    if "mean_accuracy" in summary[0]:
        return sorted(summary, key=lambda row: -float(row["mean_accuracy"]))
    return summary


def plot_metric_bars(path: Path, summary: list[dict[str, object]], title: str) -> None:
    labels = [str(row.get("model", row.get("variant"))) for row in summary]
    metrics = [
        key
        for key in ["mean_mse", "mean_psnr", "mean_ssim", "mean_accuracy", "mean_auc", "mean_brier", "mean_equivariance_error"]
        if key in summary[0]
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4))
    if len(metrics) == 1:
        axes = np.asarray([axes])
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728"]
    for axis, metric, color in zip(axes, metrics, colors):
        values = [float(row[metric]) for row in summary]
        axis.bar(labels, values, color=color)
        axis.set_title(metric.replace("mean_", "").upper())
        axis.tick_params(axis="x", rotation=30, labelsize=8)
        axis.grid(alpha=0.2)
        if metric == "mean_mse":
            axis.set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def patch_stats(patch: np.ndarray, row_pos: float, col_pos: float) -> np.ndarray:
    flat = patch.reshape(-1).astype(np.float64)
    gx = np.diff(patch, axis=1)
    gy = np.diff(patch, axis=0)
    return np.asarray(
        [
            flat.mean(),
            flat.std(),
            flat.min(),
            flat.max(),
            np.quantile(flat, 0.25),
            np.quantile(flat, 0.50),
            np.quantile(flat, 0.75),
            np.abs(gx).mean(),
            np.abs(gy).mean(),
            gx.std(),
            gy.std(),
            patch[:4, :4].mean(),
            patch[:4, 4:].mean(),
            patch[4:, :4].mean(),
            patch[4:, 4:].mean(),
            math.sin(math.pi * row_pos),
            math.cos(math.pi * row_pos),
            math.sin(math.pi * col_pos),
            math.cos(math.pi * col_pos),
            math.sin(2.0 * math.pi * row_pos),
            math.cos(2.0 * math.pi * row_pos),
            math.sin(2.0 * math.pi * col_pos),
            math.cos(2.0 * math.pi * col_pos),
            flat[::2].mean(),
            flat[1::2].mean(),
            patch[2:6, 2:6].mean(),
        ],
        dtype=np.float64,
    )


def extract_patch_table(images: np.ndarray, patch_size: int = 7) -> tuple[np.ndarray, np.ndarray, list[int]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    image_ids: list[int] = []
    size = images.shape[1]
    denom = max(size - patch_size, 1)
    for image_index, image in enumerate(images):
        for row in range(0, size, patch_size):
            for col in range(0, size, patch_size):
                patch = image[row : row + patch_size, col : col + patch_size]
                features.append(patch_stats(patch, row / denom, col / denom))
                targets.append(patch.reshape(-1))
                image_ids.append(image_index)
    return np.asarray(features), np.asarray(targets), image_ids


def fashion_like_image(label: int, seed: int, size: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size), dtype=np.float64)
    cx = size / 2 + rng.normal(0, 0.8)
    cy = size / 2 + rng.normal(0, 0.8)
    if label in {0, 2, 4, 6}:  # shirt/top silhouettes
        body = (np.abs(x - cx) < 5 + label % 3) & (y > 8) & (y < 23)
        sleeves = (np.abs(x - cx) < 10) & (y > 9) & (y < 15)
        img[body | sleeves] = 0.9
    elif label in {1, 3, 5}:  # trouser/dress silhouettes
        waist = (np.abs(x - cx) < 5) & (y > 7) & (y < 13)
        legs = ((np.abs(x - (cx - 3)) < 3) | (np.abs(x - (cx + 3)) < 3)) & (y >= 13) & (y < 25)
        skirt = (np.abs(x - cx) < (y - 7) * 0.45 + 3) & (y > 8) & (y < 25)
        img[waist | (legs if label != 3 else skirt)] = 0.9
    elif label in {7, 9}:  # shoes
        shoe = ((x - cx) / 8) ** 2 + ((y - 20) / 3) ** 2 < 1.0
        sole = (np.abs(x - cx) < 10) & (y > 21) & (y < 24)
        img[shoe | sole] = 0.9
    else:  # bag
        bag = (np.abs(x - cx) < 7) & (y > 11) & (y < 24)
        handle = ((x - cx) / 6) ** 2 + ((y - 11) / 4) ** 2 < 1.0
        img[bag | (handle & (y < 13))] = 0.85
    img += rng.normal(0.0, 0.05, size=(size, size))
    return np.clip(img, 0.0, 1.0)


def _normalize_image_array(images: np.ndarray, limit: int | None = None) -> np.ndarray:
    array = np.asarray(images)
    if limit is not None:
        array = array[:limit]
    array = array.astype(np.float64)
    if array.ndim == 4:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] in {3, 4}:
            array = array[..., :3].mean(axis=-1)
    if array.max(initial=0.0) > 1.5:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def _load_local_npz(candidates: Iterable[Path]) -> tuple[np.ndarray, np.ndarray, str] | None:
    for path in candidates:
        if not path.exists():
            continue
        data = np.load(path)
        if {"images", "labels"}.issubset(data.files):
            images = data["images"]
            labels = data["labels"]
        elif {"x_train", "y_train"}.issubset(data.files):
            images = data["x_train"]
            labels = data["y_train"]
        elif {"train_images", "train_labels"}.issubset(data.files):
            images = data["train_images"]
            labels = data["train_labels"].reshape(-1)
        else:
            continue
        return _normalize_image_array(images), labels.astype(int), f"local-npz:{path}"
    return None


def _load_medmnist_cached_dataset() -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        import medmnist
        from medmnist import INFO
    except Exception:
        return None

    preferred = [
        "pathmnist",
        "dermamnist",
        "octmnist",
        "pneumoniamnist",
        "bloodmnist",
        "tissuemnist",
        "organamnist",
    ]
    roots = [root for root in MEDMNIST_ROOT_CANDIDATES if root is not None]
    for root in roots:
        for dataset_name in preferred:
            info = INFO.get(dataset_name)
            if not info:
                continue
            dataset_class = getattr(medmnist, info["python_class"], None)
            if dataset_class is None:
                continue
            try:
                dataset = dataset_class(split="train", root=str(root), download=False, as_rgb=False, size=28)
            except Exception:
                continue
            images = getattr(dataset, "imgs", None)
            labels = getattr(dataset, "labels", None)
            if images is None or labels is None:
                continue
            images_np = np.asarray(images)
            if images_np.ndim == 4:
                images_np = images_np[..., 0]
            labels_np = np.asarray(labels).reshape(-1)
            return (
                _normalize_image_array(images_np, limit=400),
                labels_np[:400].astype(int),
                f"medmnist-cached-{dataset_name}:{root}",
            )
    return None


def _load_torchvision_cached_dataset() -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        from torchvision import datasets
    except Exception:
        return None

    dataset_classes = [
        ("fashion-mnist", datasets.FashionMNIST),
        ("mnist", datasets.MNIST),
    ]
    for root in DATASET_ROOT_CANDIDATES:
        for dataset_name, dataset_class in dataset_classes:
            try:
                dataset = dataset_class(root=str(root), train=True, download=False)
            except Exception:
                continue
            images = getattr(dataset, "data", None)
            labels = getattr(dataset, "targets", None)
            if images is None or labels is None:
                continue
            images_np = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
            labels_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
            return (
                _normalize_image_array(images_np, limit=400),
                labels_np[:400].astype(int),
                f"torchvision-cached-{dataset_name}:{root}",
            )
    return None


def load_second_dataset() -> tuple[np.ndarray, np.ndarray, str]:
    fixed_npz_candidates = [
        WORKSPACE / "datasets" / "pathmnist.npz",
        WORKSPACE / "datasets" / "dermamnist.npz",
        WORKSPACE / "datasets" / "octmnist.npz",
        WORKSPACE / "datasets" / "pneumoniamnist.npz",
        WORKSPACE / "datasets" / "bloodmnist.npz",
        Path.home() / ".medmnist" / "pathmnist.npz",
        Path.home() / ".medmnist" / "dermamnist.npz",
        Path.home() / ".medmnist" / "octmnist.npz",
        Path.home() / ".medmnist" / "pneumoniamnist.npz",
        Path.home() / ".medmnist" / "bloodmnist.npz",
        WORKSPACE / "datasets" / "fashion_mnist_subset.npz",
        WORKSPACE / "datasets" / "mnist_subset.npz",
        Path.home() / ".keras" / "datasets" / "fashion-mnist.npz",
        Path.home() / ".keras" / "datasets" / "mnist.npz",
        Path.home() / ".keras" / "datasets" / "fashion_mnist.npz",
    ]
    recursive_npz_candidates: list[Path] = []
    for root in [candidate for candidate in MEDMNIST_ROOT_CANDIDATES if candidate is not None]:
        if root.exists():
            recursive_npz_candidates.extend(sorted(root.rglob("*mnist*.npz")))
            recursive_npz_candidates.extend(sorted(root.rglob("*minist*.npz")))
    npz_candidates = fixed_npz_candidates + recursive_npz_candidates
    local_npz = _load_local_npz(npz_candidates)
    if local_npz is not None:
        return local_npz

    cached_medmnist = _load_medmnist_cached_dataset()
    if cached_medmnist is not None:
        return cached_medmnist

    cached_torchvision = _load_torchvision_cached_dataset()
    if cached_torchvision is not None:
        return cached_torchvision

    local_npz = WORKSPACE / "datasets" / "fashion_mnist_subset.npz"
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label in range(10):
        for index in range(40):
            images.append(fashion_like_image(label, seed=label * 1000 + index))
            labels.append(label)
    return np.asarray(images), np.asarray(labels), "synthetic-fashion-like-fallback"


def run_second_dataset_subset() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    images, labels, dataset_name = load_second_dataset()
    rows: list[dict[str, object]] = []
    variants = [
        ("HVK2D-pair-observable", lambda base, seed: real_newhvk_features(base)),
        ("no-entanglement", lambda base, seed: real_no_entanglement_features(base)),
        ("local-only", lambda base, seed: real_local_observables_only(base)),
        ("raw-linear", lambda base, seed: real_raw_linear_features(base)),
        ("parameter-matched", real_parameter_matched_classical_features),
        ("quadratic-classical", lambda base, seed: real_quadratic_classical_features(base)),
    ]
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        train_idx: list[int] = []
        test_idx: list[int] = []
        for label in sorted(set(labels)):
            candidates = np.where(labels == label)[0]
            order = rng.permutation(candidates)
            train_idx.extend(order[:20])
            test_idx.extend(order[20:30])
        train_images = images[train_idx]
        test_images = images[test_idx]
        x_train, y_train, _ = extract_patch_table(train_images, patch_size=7)
        x_test, y_test, patch_image_ids = extract_patch_table(test_images, patch_size=7)
        x_train, x_test = standardize_train_test(x_train, x_test)
        for model, feature_fn in variants:
            pred = ridge_fit_predict(feature_fn(x_train, seed), y_train, feature_fn(x_test, seed))
            for image_id in sorted(set(patch_image_ids)):
                mask = np.asarray(patch_image_ids) == image_id
                target = y_test[mask].reshape(4, 4, 7, 7).transpose(0, 2, 1, 3).reshape(28, 28)
                prediction = pred[mask].reshape(4, 4, 7, 7).transpose(0, 2, 1, 3).reshape(28, 28)
                metrics = image_metric_rows(np.clip(prediction, 0, 1), target)
                rows.append({"dataset": dataset_name, "seed": seed, "model": model, "image_id": image_id, **metrics})
    return rows, aggregate(rows)


def cifar_patch_table_for_paths(paths: list[Path], patch_size: int = 8) -> tuple[np.ndarray, np.ndarray, list[int]]:
    images = np.asarray([load_cifar_gray(path) for path in paths])
    return extract_patch_table(images, patch_size=patch_size)


def topology_1d_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    pairs = np.stack([local[:, i] * local[:, i + 1] for i in range(6)], axis=1)
    return np.concatenate([local, pairs, np.sin(np.pi * pairs), base[:, 18:26]], axis=1)[:, :32]


def topology_2d_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    edge_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (2, 5)]
    pairs = np.stack([local[:, i] * local[:, j] for i, j in edge_pairs], axis=1)
    return np.concatenate([local, pairs, np.sin(np.pi * pairs), base[:, 18:26]], axis=1)[:, :32]


def run_topology_matched() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    paths = sorted(CIFAR_IMAGES.glob("*.png"))
    rows: list[dict[str, object]] = []
    variants = [("hvk1d-chain-matched", topology_1d_features), ("hvk2d-grid-matched", topology_2d_features)]
    for seed in [0, 1, 2]:
        order = np.random.default_rng(seed).permutation(len(paths))
        train_paths = [paths[i] for i in order[:6]]
        test_paths = [paths[i] for i in order[6:10]]
        x_train, y_train, _ = cifar_patch_table_for_paths(train_paths)
        x_test, y_test, patch_image_ids = cifar_patch_table_for_paths(test_paths)
        x_train, x_test = standardize_train_test(x_train, x_test)
        for model, feature_fn in variants:
            pred = ridge_fit_predict(feature_fn(x_train), y_train, feature_fn(x_test))
            mse = float(np.mean((pred - y_test) ** 2))
            rows.append({"seed": seed, "model": model, "mse": mse, "psnr": psnr_from_mse(mse), "r2": r2_score(y_test, pred), "feature_dim": 32, "pair_channels": 6})
    return rows, aggregate(rows)


def d4_transforms(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": image.copy(),
        "rot90": np.rot90(image, 1),
        "rot180": np.rot90(image, 2),
        "rot270": np.rot90(image, 3),
        "flip_h": np.fliplr(image),
        "flip_v": np.flipud(image),
        "transpose": image.T,
        "anti_transpose": np.fliplr(np.flipud(image)).T,
    }


def grid_feature_map_base(image: np.ndarray, mode: str) -> np.ndarray:
    features, _, _ = extract_patch_table(np.asarray([image]), patch_size=8)
    features, _ = standardize_train_test(features, features)
    if mode == "HVK2D-positional":
        mapped = real_newhvk_features(features)
    elif mode == "no-positional":
        stripped = features.copy()
        stripped[:, 18:26] = 0.0
        mapped = real_newhvk_features(stripped)
    else:
        mapped = real_local_observables_only(features)
    return mapped.reshape(4, 4, -1)


def inverse_transform_grid(grid: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return grid.copy()
    if transform == "rot90":
        return np.rot90(grid, -1, axes=(0, 1))
    if transform == "rot180":
        return np.rot90(grid, 2, axes=(0, 1))
    if transform == "rot270":
        return np.rot90(grid, 1, axes=(0, 1))
    if transform == "flip_h":
        return np.fliplr(grid)
    if transform == "flip_v":
        return np.flipud(grid)
    if transform == "transpose":
        return np.swapaxes(grid, 0, 1)
    if transform == "anti_transpose":
        return np.flipud(np.fliplr(np.swapaxes(grid, 0, 1)))
    raise ValueError(transform)


def d4_pooled_feature_map(image: np.ndarray) -> np.ndarray:
    aligned = [
        inverse_transform_grid(grid_feature_map_base(transformed, "HVK2D-positional"), transform)
        for transform, transformed in d4_transforms(image).items()
    ]
    return np.mean(aligned, axis=0)


def grid_feature_map(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "D4-pooled-HVK2D":
        return d4_pooled_feature_map(image)
    return grid_feature_map_base(image, mode)


def run_d4_equivariance() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    modes = ["D4-pooled-HVK2D", "HVK2D-positional", "no-positional", "local-raw"]
    for image_path in sorted(CIFAR_IMAGES.glob("*.png")):
        image = load_cifar_gray(image_path)
        for mode in modes:
            base = grid_feature_map(image, mode)
            denom = float(np.linalg.norm(base) + 1e-8)
            for name, transformed_image in d4_transforms(image).items():
                if name == "identity":
                    continue
                transformed = inverse_transform_grid(grid_feature_map(transformed_image, mode), name)
                err = float(np.linalg.norm(base - transformed) / denom)
                rows.append({"image": image_path.name, "model": mode, "transform": name, "equivariance_error": err})
    return rows, aggregate(rows)


def snake_indices(size: int) -> list[int]:
    idx: list[int] = []
    for row in range(size):
        cols = range(size) if row % 2 == 0 else range(size - 1, -1, -1)
        idx.extend(row * size + col for col in cols)
    return idx


def hilbert_indices(size: int) -> list[int]:
    def xy_to_d(n: int, x: int, y: int) -> int:
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if x & s else 0
            ry = 1 if y & s else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                x, y = y, x
            s //= 2
        return d
    coords = [(xy_to_d(size, col, row), row * size + col) for row in range(size) for col in range(size)]
    return [idx for _, idx in sorted(coords)]


def ordering_features(patch: np.ndarray, order: str, n_features: int = 16) -> np.ndarray:
    size = patch.shape[0]
    if order == "raster":
        indices = list(range(size * size))
    elif order == "snake":
        indices = snake_indices(size)
    else:
        indices = hilbert_indices(size)
    vector = patch.reshape(-1)[indices]
    chunks = np.array_split(vector, n_features)
    return np.asarray([chunk.mean() for chunk in chunks] + [chunk.std() for chunk in chunks], dtype=np.float64)


def run_mps_ordering_ablation() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    paths = sorted(CIFAR_IMAGES.glob("*.png"))
    rows: list[dict[str, object]] = []
    for seed in [0, 1, 2]:
        order = np.random.default_rng(seed).permutation(len(paths))
        train_paths = [paths[i] for i in order[:6]]
        test_paths = [paths[i] for i in order[6:10]]
        train_images = [load_cifar_gray(path) for path in train_paths]
        test_images = [load_cifar_gray(path) for path in test_paths]
        for ordering in ["raster", "snake", "hilbert"]:
            x_train: list[np.ndarray] = []
            y_train: list[np.ndarray] = []
            x_test: list[np.ndarray] = []
            y_test: list[np.ndarray] = []
            for bucket_x, bucket_y, images in [(x_train, y_train, train_images), (x_test, y_test, test_images)]:
                for image in images:
                    for row in range(0, 32, 8):
                        for col in range(0, 32, 8):
                            patch = image[row : row + 8, col : col + 8]
                            bucket_x.append(ordering_features(patch, ordering))
                            bucket_y.append(patch.reshape(-1))
            train_x, test_x = standardize_train_test(np.asarray(x_train), np.asarray(x_test))
            pred = ridge_fit_predict(train_x, np.asarray(y_train), test_x)
            mse = float(np.mean((pred - np.asarray(y_test)) ** 2))
            rows.append({"seed": seed, "model": ordering, "mse": mse, "psnr": psnr_from_mse(mse), "r2": r2_score(np.asarray(y_test), pred)})
    return rows, aggregate(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def logistic_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    test_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    weights = np.zeros(x_aug.shape[1], dtype=np.float64)
    y = y_train.astype(np.float64)
    lr = 0.08
    reg = 1e-3
    for _ in range(900):
        probs = sigmoid(x_aug @ weights)
        grad = (x_aug.T @ (probs - y)) / x_aug.shape[0] + reg * weights
        grad[-1] -= reg * weights[-1]
        weights -= lr * grad
    return sigmoid(test_aug @ weights)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    y = y_true.astype(int)
    accuracy = float(np.mean(predictions == y))
    tp = float(np.sum((predictions == 1) & (y == 1)))
    tn = float(np.sum((predictions == 0) & (y == 0)))
    fp = float(np.sum((predictions == 1) & (y == 0)))
    fn = float(np.sum((predictions == 0) & (y == 1)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    brier = float(np.mean((probabilities - y) ** 2))
    order = np.argsort(probabilities)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    positives = y == 1
    n_pos = int(np.sum(positives))
    n_neg = int(len(y) - n_pos)
    if n_pos and n_neg:
        auc = (float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        auc = float("nan")
    return {
        "accuracy": accuracy,
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
        "specificity": float(specificity),
        "brier": brier,
    }


def wisconsin_hvk_features(base: np.ndarray) -> np.ndarray:
    local = select_same_width(base, 18)
    pairs = np.stack(
        [
            local[:, 0] * local[:, 1],
            local[:, 2] * local[:, 3],
            local[:, 4] * local[:, 5],
            local[:, 6] * local[:, 7],
            local[:, 8] * local[:, 9],
            local[:, 10] * local[:, 11],
            local[:, 12] * local[:, 13],
            local[:, 14] * local[:, 15],
        ],
        axis=1,
    )
    return select_same_width(np.concatenate([local, pairs, np.sin(np.pi * pairs)], axis=1), 32)


def wisconsin_no_entanglement_features(base: np.ndarray) -> np.ndarray:
    local = select_same_width(base, 18)
    return select_same_width(np.concatenate([local, np.sin(np.pi * local[:, :10]), np.cos(np.pi * local[:, :4])], axis=1), 32)


def wisconsin_rff_features(base: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(130_000 + seed)
    weights = rng.normal(0.0, 1.0 / math.sqrt(base.shape[1]), size=(base.shape[1], 32))
    bias = rng.uniform(-math.pi, math.pi, size=(32,))
    return np.sin(base @ weights + bias)


def wisconsin_quadratic_features(base: np.ndarray) -> np.ndarray:
    local = select_same_width(base, 18)
    products = []
    for index in range(0, 18, 2):
        products.append((local[:, index] * local[:, index + 1])[:, None])
    products_arr = np.concatenate(products, axis=1)
    return select_same_width(np.concatenate([local, products_arr, np.sin(np.pi * products_arr)], axis=1), 32)


def run_wisconsin_breast_cancer() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        from sklearn.datasets import load_breast_cancer
    except Exception as exc:
        rows = [
            {
                "dataset": "wisconsin-breast-cancer-unavailable",
                "seed": 0,
                "model": "loader",
                "accuracy": 0.0,
                "auc": 0.0,
                "f1": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "brier": 1.0,
                "notes": f"sklearn unavailable: {exc}",
            }
        ]
        return rows, aggregate(rows)
    data = load_breast_cancer()
    x_all = np.asarray(data.data, dtype=np.float64)
    y_all = np.asarray(data.target, dtype=int)
    rows: list[dict[str, object]] = []
    variants = [
        ("HVK2D-pair-observable", lambda base, seed: wisconsin_hvk_features(base)),
        ("no-entanglement", lambda base, seed: wisconsin_no_entanglement_features(base)),
        ("raw-linear", lambda base, seed: select_same_width(base, 32)),
        ("quadratic-classical", lambda base, seed: wisconsin_quadratic_features(base)),
        ("strict-classical-rff", wisconsin_rff_features),
    ]
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        train_idx: list[int] = []
        test_idx: list[int] = []
        for label in [0, 1]:
            label_idx = np.where(y_all == label)[0]
            order = rng.permutation(label_idx)
            cutoff = int(0.7 * len(order))
            train_idx.extend(order[:cutoff])
            test_idx.extend(order[cutoff:])
        x_train_raw = x_all[train_idx]
        x_test_raw = x_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        x_train, x_test = standardize_train_test(x_train_raw, x_test_raw)
        for model, feature_fn in variants:
            probabilities = logistic_fit_predict(feature_fn(x_train, seed), y_train, feature_fn(x_test, seed))
            rows.append(
                {
                    "dataset": "sklearn-wisconsin-breast-cancer",
                    "seed": seed,
                    "model": model,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **binary_metrics(y_test, probabilities),
                }
            )
    return rows, aggregate(rows)


def write_suite(name: str, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    result_dir = OUT / name
    result_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(result_dir / f"{name}.csv", rows)
    write_dict_csv(result_dir / f"{name}_summary.csv", summary)
    (result_dir / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (result_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_metric_bars(result_dir / f"{name}_summary.png", summary, name.replace("_", " ").title())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suites = {
        "second_dataset_subset": run_second_dataset_subset,
        "wisconsin_breast_cancer": run_wisconsin_breast_cancer,
        "topology_matched_1d2d": run_topology_matched,
        "d4_equivariance": run_d4_equivariance,
        "mps_ordering_ablation": run_mps_ordering_ablation,
    }
    manifest: dict[str, object] = {"claim_boundary": "Lightweight diagnostics; not full retraining evidence."}
    for name, fn in suites.items():
        rows, summary = fn()
        write_suite(name, rows, summary)
        manifest[name] = summary
    readme = """# Extended validation diagnostics

Generated by `main2/newHVK/run_extended_validation.py`. The path name is
historical; the reported models are only HVK1D/HVK2D validation variants.

These are lightweight, reproducible diagnostics for reviewer-requested checks
that were not present in the original runner:

- second dataset subset: uses a no-download dataset loader. It first checks
  local NPZ files such as `main2/newHVK/datasets/pathmnist.npz`,
  `dermamnist.npz`, `fashion_mnist_subset.npz`, or `mnist_subset.npz`; then
  recursively checks `medmnist` and `medminist` folders; then cached MedMNIST
  package roots (`$MEDMNIST_ROOT`, `~/.medmnist`) with `download=False`; then
  cached `torchvision` MNIST/Fashion-MNIST roots and Keras-style NPZ files. If
  no real cache is present, it uses a deterministic 10-class fashion-like
  silhouette fallback and labels it as such in the CSV/JSON `dataset` column.
- topology-matched 1D versus 2D: compares chain and grid pair channels with the
  same latent width and same number of pair channels on cached CIFAR images.
- Wisconsin breast cancer: tabular cross-domain diagnostic using the
  scikit-learn Wisconsin Breast Cancer dataset when available. This is not image
  reconstruction evidence; it tests whether the same pair-observable feature
  budget remains competitive against local/raw/quadratic controls on a small
  medical tabular benchmark.
- D4 equivariance: measures how much patch-grid features change under rotations
  and flips after undoing the corresponding patch permutation. The
  `D4-pooled-HVK2D` variant explicitly averages the HVK2D observable grid over
  all eight square symmetries after inverse-aligning each transformed feature
  grid, making the reported D4 behavior an architectural constraint rather than
  a post-hoc claim.
- MPS ordering ablation: compares raster, snake, and Hilbert-style flattening
  orders using equal-width ordered patch summaries.

Claim boundary: these results are not a substitute for full retraining on a
downloaded benchmark dataset. The D4-pooled HVK2D variant supports a real equivariance
claim for the pooled feature map, while the unpooled positional map should still
be described as non-equivariant in the present tests.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Extended validation complete: {OUT}")


if __name__ == "__main__":
    main()
