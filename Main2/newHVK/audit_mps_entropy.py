"""Generate the corrected MPS entropy audit from cached image patches.

This is the entropy stage of the original audit_review.py that generated the
published CSV on 2026-10-30. The image selection, preprocessing, MPS compression,
and both entropy calculations are unchanged. The optional comparison with a
legacy CSV is skipped when that separate file is unavailable.

Original audit_review.py SHA-256:
2c0c65a401c7e2afa80defbbab8204e2a0dd29acbeac3520f0b2fafec4ad6837
Retained mps_entropy_audit.csv SHA-256:
a3f011a25905611ef688ca292c4bb45441da22be387822c210ec684b2963bb98

Quimb 1.14.0 schmidt_values returns squared singular values, so the corrected
von Neumann entropy normalizes them directly; it does not square them again.
The legacy_feature_value_nats column preserves the historical feature formula.

Requires cached CIFAR images and Main2/newHVK/datasets/{pathmnist,bloodmnist,
pneumoniamnist}.npz. Uses no downloads, model training or quantum execution.
The retained CSV was copied without running this generator for publication.

To regenerate only when wanted:
    .venv/bin/python Main2/newHVK/audit_mps_entropy.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import quimb
import quimb.tensor as qtn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Main2" / "newHVK" / "results" / "table3_mps_entropy_ranges"


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std_ddof0": float(a.std()),
    }


def entropy_audit():
    # Read only cached images, following run_full_dataset_sameset's preprocessing.
    # No download-enabled dataset constructors or quantum circuit are called.
    image_dir = ROOT / "Baselines/cifar10_comparisons/datasets/images"
    paths = sorted(image_dir.glob("*.png"))[:3]
    assert len(paths) == 3
    datasets = {
        "cifar10": [
            cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0, (32, 32)) for p in paths
        ]
    }
    image_inputs = [{"path": str(p.relative_to(ROOT)), "sha256": checksum(p)} for p in paths]
    for name, count in [("pathmnist", 3), ("bloodmnist", 2), ("pneumoniamnist", 2)]:
        path = ROOT / f"Main2/newHVK/datasets/{name}.npz"
        with np.load(path) as dataset:
            images = dataset["train_images"][:count]
        if images.ndim == 4:
            images = images.mean(axis=-1)
        converted = []
        for image in images:
            a = np.asarray(image, dtype=np.float32)
            if a.max() > 1.5:
                a /= 255.0
            converted.append(cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA))
        datasets[name] = converted
        image_inputs.append({"path": str(path.relative_to(ROOT)), "sha256": checksum(path)})
    rows = []
    for name, images in datasets.items():
        for image_index, image in enumerate(images):
            for patch_index, (i, j) in enumerate((i, j) for i in range(0, 32, 8) for j in range(0, 32, 8)):
                vec = (image[i : i + 8, j : j + 8] + 1e-4).flatten()
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                mps = qtn.MatrixProductState.from_dense(vec.reshape([2] * 6))
                mps.compress(max_bond=4)
                mps.normalize()
                for bond in range(1, 6):
                    probabilities = np.asarray(mps.schmidt_values(bond))
                    # Quimb returns squared singular values, i.e. probabilities.
                    p = probabilities / probabilities.sum()
                    p = p[p > 0]
                    entropy = float(-np.sum(p * np.log(p)))
                    legacy = probabilities**2
                    legacy = legacy / (legacy.sum() + 1e-8)
                    legacy_entropy = float(np.float32(-np.sum(legacy * np.log(legacy + 1e-8))))
                    rows.append(
                        {
                            "group": "CIFAR-10" if name == "cifar10" else "MedMNIST",
                            "dataset": name,
                            "image_index": image_index,
                            "patch_index": patch_index,
                            "bond_index": bond,
                            "von_neumann_entropy_nats": entropy,
                            "legacy_feature_value_nats": legacy_entropy,
                        }
                    )
    assert len(rows) == 800
    # The historical comparison file is optional: it was not part of main
    # when this standalone generator was published. It does not enter the
    # corrected entropy calculation or any column of the output CSV.
    legacy_path = ROOT / "Main2/newHVK/results/table3_mps_entropy_ranges/mps_bipartite_entropies.csv"
    max_error = None
    if legacy_path.exists():
        with legacy_path.open() as stream:
            retained = list(csv.DictReader(stream))
        assert len(retained) == len(rows)
        max_error = 0.0
        for before, now in zip(retained, rows, strict=True):
            assert before["dataset"] == now["dataset"]
            for key in ["image_index", "patch_index", "bond_index"]:
                assert int(before[key]) == now[key]
            max_error = max(max_error, abs(float(before["entropy_nats"]) - now["legacy_feature_value_nats"]))
        assert max_error < 2e-6, max_error
    write_csv("mps_entropy_audit.csv", rows)
    result = {
        "quimb_version": quimb.__version__,
        "units": "nats",
        "quantity": "von Neumann entropy of the normalized compressed MPS",
        "historical_code_issue": "schmidt_values already squares singular values; the feature code squares again",
        "max_legacy_reproduction_absolute_error": max_error,
        "protocol": {
            "patch_size": 8,
            "stride": 8,
            "sites": 6,
            "bond_dimension": 4,
            "input_offset": 0.0001,
            "n_images": 10,
            "n_patches": 160,
            "n_bonds": 800,
        },
        "inputs": image_inputs,
        "groups": {},
        "datasets": {},
    }
    for group in ["CIFAR-10", "MedMNIST"]:
        result["groups"][group] = summary([r["von_neumann_entropy_nats"] for r in rows if r["group"] == group])
    for name in datasets:
        result["datasets"][name] = summary([r["von_neumann_entropy_nats"] for r in rows if r["dataset"] == name])
    (OUT / "mps_entropy_corrected_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = entropy_audit()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
