"""Position-only control for the same-set reconstruction table.

This matches ``run_full_dataset_sameset.py`` except that the PCAE2D observable
vector is fixed to zero.  A fresh PCAE2D decoder is trained per image using only
the four-dimensional positional encoding.  The experiment therefore tests how
much of the per-image reconstruction can be memorised from patch position alone.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "Main", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Main2.newHVK.run_full_dataset_sameset import (  # noqa: E402
    DATASET_LOADERS,
    IMAGE_SIZE,
    LR,
    PATCH_SIZE,
    PATCH_STRIDE,
    POSITIONAL_DIM,
    extract_patches,
    psnr_ssim,
)
from Main2.src.model import OBS_DIM, PatchDecoder  # noqa: E402
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding  # noqa: E402
from src.training.training import resolve_device  # noqa: E402

OUTPUT_DIR = ROOT / "Main2" / "newHVK" / "results" / "position_only_sameset"
IMAGE_COUNTS = {
    "cifar10": 3,
    "mnist": 3,
    "fashion-mnist": 3,
    "pathmnist": 3,
    "bloodmnist": 2,
    "pneumoniamnist": 2,
}
SEED = 0
STEPS = 100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(image: np.ndarray, device: torch.device) -> dict[str, float | int]:
    seed_everything(SEED)
    patches, raw_positions = extract_patches(image, PATCH_SIZE, PATCH_STRIDE)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM).to(device)
    targets = torch.tensor(patches, dtype=torch.float32, device=device).unsqueeze(1)
    zero_observables = torch.zeros((len(patches), OBS_DIM), dtype=torch.float32, device=device)

    decoder = PatchDecoder(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=LR)

    started = time.perf_counter()
    for _ in range(STEPS):
        decoder.train()
        optimizer.zero_grad()
        prediction = decoder(zero_observables, positions)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()
        optimizer.step()

    decoder.eval()
    with torch.no_grad():
        predicted_patches = decoder(zero_observables, positions).cpu().numpy()[:, 0]

    reconstruction = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for patch, (row, col) in zip(predicted_patches, raw_positions, strict=True):
        i = int(round(row * IMAGE_SIZE))
        j = int(round(col * IMAGE_SIZE))
        reconstruction[i : i + PATCH_SIZE, j : j + PATCH_SIZE] = patch

    return {
        **psnr_ssim(reconstruction, image),
        "steps": STEPS,
        "seed": SEED,
        "n_patches": len(patches),
        "decoder_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    device = resolve_device("auto")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}", flush=True)

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for dataset, count in IMAGE_COUNTS.items():
        print(f"\n=== {dataset} (n={count}) ===", flush=True)
        images = DATASET_LOADERS[dataset](count)
        for image_index, image in enumerate(images):
            metrics = train_one(image, device)
            row: dict[str, object] = {
                "dataset": dataset,
                "image_index": image_index,
                **metrics,
            }
            rows.append(row)
            print(
                f"  image {image_index}: PSNR={metrics['psnr']:.2f} dB, "
                f"SSIM={metrics['ssim']:.4f}, time={metrics['elapsed_seconds']:.2f}s",
                flush=True,
            )

    summary = []
    for dataset in IMAGE_COUNTS:
        selected = [row for row in rows if row["dataset"] == dataset]
        psnrs = np.asarray([row["psnr"] for row in selected], dtype=float)
        ssims = np.asarray([row["ssim"] for row in selected], dtype=float)
        summary.append(
            {
                "dataset": dataset,
                "n_images": len(selected),
                "mean_psnr": float(psnrs.mean()),
                "std_psnr": float(psnrs.std(ddof=0)),
                "mean_ssim": float(ssims.mean()),
                "std_ssim": float(ssims.std(ddof=0)),
            }
        )

    result = {
        "control": "position-only; 19 PCAE2D observable channels fixed to zero",
        "seed": SEED,
        "steps_per_image": STEPS,
        "learning_rate": LR,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "positional_dim": POSITIONAL_DIM,
        "observable_dim": OBS_DIM,
        "decoder_parameters": rows[0]["decoder_parameters"],
        "total_elapsed_seconds": time.perf_counter() - started,
        "runs": rows,
        "summary": summary,
    }
    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Position-only summary ===")
    for row in summary:
        print(
            f"{row['dataset']}: n={row['n_images']} "
            f"PSNR={row['mean_psnr']:.2f}+/-{row['std_psnr']:.2f} dB, "
            f"SSIM={row['mean_ssim']:.4f}+/-{row['std_ssim']:.4f}"
        )
    print(f"Total time: {result['total_elapsed_seconds']:.2f}s")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
