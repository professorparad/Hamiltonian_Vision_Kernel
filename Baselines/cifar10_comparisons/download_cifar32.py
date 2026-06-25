"""
Download CIFAR-10 subset at native 32×32 resolution (no resizing to 256).
Images are saved at their original 32×32 size for CIFAR-adapted HVK benchmarks.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "datasets"
DEFAULT_CACHE_DIR = REPO_ROOT / "Baselines" / "datasets" / "_cache"


def unpickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def download_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "cifar-10-python.tar.gz"
    if archive_path.exists():
        return archive_path
    urllib.request.urlretrieve(CIFAR10_URL, archive_path)
    return archive_path


def extract_archive(archive_path: Path, cache_dir: Path) -> Path:
    extracted_root = cache_dir / "cifar-10-batches-py"
    if extracted_root.exists():
        return extracted_root
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(cache_dir)
    return extracted_root


def save_subset(
    extracted_root: Path,
    output_dir: Path,
    count: int,
    split: str,
) -> dict:
    """Save CIFAR-10 images at native 32×32 — no resizing."""
    batch_path = extracted_root / ("test_batch" if split == "test" else "data_batch_1")
    batch = unpickle(batch_path)
    data = batch["data"]
    labels = batch["labels"]
    filenames = batch["filenames"]

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index in range(min(count, len(data))):
        rgb = data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # NO RESIZE — keep at 32×32
        label = int(labels[index])
        stem = Path(filenames[index]).stem
        image_name = f"{index:04d}_{CLASS_NAMES[label]}_{stem}.png"
        image_path = images_dir / image_name
        cv2.imwrite(str(image_path), gray)
        rows.append({
            "index": index,
            "source_filename": filenames[index],
            "class_id": label,
            "class_name": CLASS_NAMES[label],
            "image_path": str(image_path.relative_to(output_dir)),
        })

    manifest = {
        "dataset": "CIFAR-10 subset (native 32×32)",
        "source_url": CIFAR10_URL,
        "source_reference": "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009",
        "split": split,
        "count": len(rows),
        "original_image_size": 32,
        "saved_image_size": 32,
        "color_mode": "grayscale (native res)",
        "images_dir": str(images_dir),
        "samples": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 subset at native 32×32.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = download_archive(args.cache_dir)
    extracted_root = extract_archive(archive_path, args.cache_dir)
    manifest = save_subset(
        extracted_root=extracted_root,
        output_dir=args.output_dir,
        count=args.count,
        split=args.split,
    )
    print(f"Saved {manifest['count']} CIFAR-10 images at 32×32 to {args.output_dir / 'images'}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
