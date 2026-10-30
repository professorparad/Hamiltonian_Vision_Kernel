"""Compute MPS bipartite-entropy ranges for the natural-image table patches.

This reuses the loaders, patching, preprocessing, and ``extract_mps_features``
configuration from ``run_full_dataset_sameset.py``.  No model training is needed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "Main"
SCRIPT_DIR = ROOT / "Main2" / "newHVK"
for path in (MAIN_DIR, SCRIPT_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_full_dataset_sameset import (  # noqa: E402
    DATASET_LOADERS,
    extract_patches,
    PATCH_SIZE,
    PATCH_STRIDE,
)
from src.tensornetworks.mps_features import extract_mps_features  # noqa: E402


N_SITES = 6
BOND_DIM = 4
ENTROPY_START = 2 * N_SITES + (N_SITES - 1)
DATASET_IMAGE_COUNTS = {
    "cifar10": 3,
    "pathmnist": 3,
    "bloodmnist": 2,
    "pneumoniamnist": 2,
}
OUTPUT_DIR = (
    ROOT / "Main2" / "newHVK" / "results" / "table3_mps_entropy_ranges"
)


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n_entropy_values": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "protocol": {
            "source_table_script": "Main2/newHVK/run_full_dataset_sameset.py",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "patches_per_image": 16,
            "n_sites": N_SITES,
            "bond_dim": BOND_DIM,
            "input_offset": 1e-4,
            "entropy_log_base": "natural",
            "range_definition": "minimum and maximum over every patch and MPS bond",
        },
        "datasets": {},
    }

    for dataset, n_images in DATASET_IMAGE_COUNTS.items():
        images = DATASET_LOADERS[dataset](n_images)
        dataset_values: list[float] = []
        for image_index, image in enumerate(images):
            patches, _ = extract_patches(image, PATCH_SIZE, PATCH_STRIDE)
            for patch_index, patch in enumerate(patches):
                features = extract_mps_features(
                    patch + 1e-4,
                    n_sites=N_SITES,
                    bond_dim=BOND_DIM,
                )
                entropies = features[ENTROPY_START:]
                if entropies.shape != (N_SITES - 1,):
                    raise RuntimeError(
                        f"Unexpected entropy shape {entropies.shape} for {dataset}"
                    )
                for bond_index, entropy in enumerate(entropies, start=1):
                    value = float(entropy)
                    dataset_values.append(value)
                    rows.append(
                        {
                            "group": "CIFAR-10" if dataset == "cifar10" else "MedMNIST",
                            "dataset": dataset,
                            "image_index": image_index,
                            "patch_index": patch_index,
                            "bond_index": bond_index,
                            "entropy_nats": value,
                        }
                    )
        summary["datasets"][dataset] = {
            "n_images": n_images,
            "n_patches": n_images * 16,
            **summarize(dataset_values),
        }

    for group in ("CIFAR-10", "MedMNIST"):
        group_values = [
            float(row["entropy_nats"]) for row in rows if row["group"] == group
        ]
        group_rows = [row for row in rows if row["group"] == group]
        summary.setdefault("groups", {})[group] = {
            "n_images": sum(
                DATASET_IMAGE_COUNTS[name]
                for name in DATASET_IMAGE_COUNTS
                if (name == "cifar10") == (group == "CIFAR-10")
            ),
            "n_patches": len(group_rows) // (N_SITES - 1),
            **summarize(group_values),
        }

    all_values = [float(row["entropy_nats"]) for row in rows]
    summary["combined"] = {
        "n_images": sum(DATASET_IMAGE_COUNTS.values()),
        "n_patches": len(rows) // (N_SITES - 1),
        **summarize(all_values),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "mps_bipartite_entropies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
