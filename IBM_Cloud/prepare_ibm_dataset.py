from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"
MONALISA_PATH = REPO_ROOT / "Main" / "data" / "monalisa.jpg"
CIFAR_DIR = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "datasets" / "images"


def load_grayscale(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def extract_patch_vectors(image: np.ndarray, patch_size: int, max_patches: int) -> np.ndarray:
    patches = []
    for row in range(0, image.shape[0], patch_size):
        for col in range(0, image.shape[1], patch_size):
            patch = image[row : row + patch_size, col : col + patch_size]
            vector = patch.reshape(-1).astype(np.float32)
            norm = np.linalg.norm(vector) + 1e-8
            patches.append(vector / norm)
            if len(patches) >= max_patches:
                return np.asarray(patches, dtype=np.float32)
    return np.asarray(patches, dtype=np.float32)


def build_monalisa(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    image = load_grayscale(MONALISA_PATH, args.image_size)
    vectors = extract_patch_vectors(image, args.patch_size, args.max_patches)
    metadata = {
        "source": "monalisa",
        "source_path": str(MONALISA_PATH),
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "max_patches": args.max_patches,
    }
    return vectors, metadata


def build_cifar(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    paths = sorted(CIFAR_DIR.glob("*.png"))
    if not paths:
        raise FileNotFoundError(
            f"No CIFAR PNG files found in {CIFAR_DIR}. Run Baselines/cifar10_comparisons/download_cifar32.py first."
        )
    vectors = []
    used = []
    for path in paths:
        image = load_grayscale(path, args.image_size)
        image_vectors = extract_patch_vectors(image, args.patch_size, args.max_patches - len(vectors))
        vectors.extend(image_vectors)
        used.append(str(path))
        if len(vectors) >= args.max_patches:
            break
    metadata = {
        "source": "cifar",
        "source_paths": used,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "max_patches": args.max_patches,
    }
    return np.asarray(vectors, dtype=np.float32), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tiny image patches for IBM Quantum HVK probes.")
    parser.add_argument("--source", choices=["monalisa", "cifar"], default="monalisa")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size % args.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vectors, metadata = build_monalisa(args) if args.source == "monalisa" else build_cifar(args)
    output_path = args.output_dir / f"{args.source}_patches.npz"
    np.savez_compressed(output_path, patch_vectors=vectors, metadata=json.dumps(metadata, indent=2))
    print(f"Saved {len(vectors)} patches to {output_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
