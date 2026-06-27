from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(__file__).resolve().parent / "outputs" / "ibm_epoch_probe_results.json"
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "monalisa_original_vs_heron_patch_proxy.png"


def load_metadata(dataset_path: Path) -> dict:
    data = np.load(dataset_path, allow_pickle=False)
    if "metadata" not in data:
        return {"image_size": 64, "patch_size": 8}
    return json.loads(str(data["metadata"]))


def load_original(metadata: dict) -> np.ndarray:
    source_path = Path(metadata.get("source_path", REPO_ROOT / "Main" / "data" / "monalisa.jpg"))
    image_size = int(metadata.get("image_size", 64))
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Original image not found: {source_path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def build_patch_image(rows: list[dict], image_size: int, patch_size: int) -> np.ndarray:
    grid = image_size // patch_size
    patch_values = np.full((grid, grid), np.nan, dtype=np.float32)
    for row in rows:
        patch_index = int(row["patch_index"])
        patch_row, patch_col = divmod(patch_index, grid)
        if patch_row < grid and patch_col < grid:
            patch_values[patch_row, patch_col] = float(row["hardware_proxy_loss"])

    if np.isnan(patch_values).any():
        raise ValueError(
            "Hardware results do not cover every patch. Run with --max-patches equal to "
            f"{grid * grid} for a full {grid}x{grid} patch map."
        )

    min_value = float(patch_values.min())
    max_value = float(patch_values.max())
    normalized = (patch_values - min_value) / (max_value - min_value + 1e-8)
    proxy_brightness = 1.0 - normalized
    return np.kron(proxy_brightness, np.ones((patch_size, patch_size), dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot original Mona Lisa beside real IBM Heron patch proxy map.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--variant", default="hvk2d")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    rows = [row for row in rows if row["variant"] == args.variant and int(row["epoch"]) == args.epoch]
    if not rows:
        raise ValueError(f"No rows found for variant={args.variant!r}, epoch={args.epoch}.")

    metadata = load_metadata(args.dataset)
    image_size = int(metadata.get("image_size", 64))
    patch_size = int(metadata.get("patch_size", 8))
    original = load_original(metadata)
    hardware_proxy = build_patch_image(rows, image_size, patch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(hardware_proxy, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Heron {args.variant} proxy, epoch {args.epoch}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved hardware proxy comparison: {args.output}")


if __name__ == "__main__":
    main()
