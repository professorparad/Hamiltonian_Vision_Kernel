from __future__ import annotations

import argparse
import csv
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "Baselines" / "datasets" / "cifar10_subset"
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
    image_size: int,
    split: str,
) -> dict:
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
        resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        label = int(labels[index])
        stem = Path(filenames[index]).stem
        image_name = f"{index:04d}_{CLASS_NAMES[label]}_{stem}.png"
        image_path = images_dir / image_name
        cv2.imwrite(str(image_path), resized)
        rows.append(
            {
                "index": index,
                "source_filename": filenames[index],
                "class_id": label,
                "class_name": CLASS_NAMES[label],
                "image_path": str(image_path.relative_to(output_dir)),
            }
        )

    manifest = {
        "dataset": "CIFAR-10 subset",
        "source_url": CIFAR10_URL,
        "source_reference": "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009",
        "split": split,
        "count": len(rows),
        "original_image_size": 32,
        "saved_image_size": image_size,
        "color_mode": "grayscale",
        "images_dir": str(images_dir),
        "samples": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def save_csv_subset(
    csv_root: Path,
    output_dir: Path,
    count: int,
    image_size: int,
    split: str,
) -> dict:
    csv_path = csv_root / ("test.csv" if split == "test" else "train.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CIFAR CSV split: {csv_path}")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 3072:
            raise ValueError(f"Expected at least 3072 pixel columns in {csv_path}")

        for index, row in enumerate(reader):
            if index >= count:
                break
            pixels = np.asarray(row[:3072], dtype=np.uint8)
            rgb = pixels.reshape(3, 32, 32).transpose(1, 2, 0)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(
                gray, (image_size, image_size), interpolation=cv2.INTER_CUBIC
            )
            image_name = f"{index:04d}_cifar10_{split}.png"
            image_path = images_dir / image_name
            cv2.imwrite(str(image_path), resized)
            rows.append(
                {
                    "index": index,
                    "source_row": index,
                    "class_id": None,
                    "class_name": None,
                    "image_path": str(image_path.relative_to(output_dir)),
                }
            )

    manifest = {
        "dataset": "CIFAR-10 CSV subset",
        "source_path": str(csv_root),
        "source_reference": "Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009",
        "split": split,
        "count": len(rows),
        "original_image_size": 32,
        "saved_image_size": image_size,
        "color_mode": "grayscale",
        "images_dir": str(images_dir),
        "samples": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and save a CIFAR-10 subset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--source-csv-dir",
        type=Path,
        help="Use an existing CSV-format CIFAR-10 directory containing train.csv/test.csv.",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_csv_dir:
        manifest = save_csv_subset(
            csv_root=args.source_csv_dir,
            output_dir=args.output_dir,
            count=args.count,
            image_size=args.image_size,
            split=args.split,
        )
    else:
        archive_path = download_archive(args.cache_dir)
        extracted_root = extract_archive(archive_path, args.cache_dir)
        manifest = save_subset(
            extracted_root=extracted_root,
            output_dir=args.output_dir,
            count=args.count,
            image_size=args.image_size,
            split=args.split,
        )
    print(f"Saved {manifest['count']} CIFAR-10 images to {args.output_dir / 'images'}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
