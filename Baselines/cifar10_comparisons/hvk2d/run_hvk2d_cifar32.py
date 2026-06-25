"""
HVK2D on CIFAR-10 at native 32×32 resolution (patch_size=8, n_sites=6).
Uses the 2D grid quantum model from Main2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

MAIN_DIR = BENCH_ROOT.parents[1] / "Main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
REPO_ROOT = BENCH_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from common import (
    DEFAULT_DATASET_DIR,
    compute_metrics,
    image_paths,
    load_grayscale_image,
    quantum_order_summary,
    save_order_parameter_plot,
    save_reconstruction_comparison,
    seed_everything,
    write_csv,
)
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from Main2.src.model import Quantum2DGridModel, PatchDecoder as PatchDecoder2D
from src.reconstruction.patch_stitching import stictch_patches
from src.reconstruction.seam_bleading import blend_seams
from src.training.training import resolve_device

CIFAR_IMAGE_SIZE = 32
CIFAR_PATCH_SIZE = 8
CIFAR_N_SITES = 6
CIFAR_POSITIONAL_DIM = 4
CIFAR_EPOCHS = 200
CIFAR_LR = 0.004


def train_hvk2d(
    image: np.ndarray, device: torch.device, epochs: int
) -> tuple[np.ndarray, list[dict]]:
    """Train HVK2D on a 32×32 CIFAR image and return reconstruction."""
    patches, raw_positions = extract_patches(image, patch_size=CIFAR_PATCH_SIZE)
    features = np.array([
        extract_mps_features(p, n_sites=CIFAR_N_SITES, bond_dim=4)
        for p in patches
    ])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (
        features_t.std(dim=0, unbiased=False) + 1e-8
    )
    positions = sinusoidal_positional_encoding(raw_positions, d_model=CIFAR_POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t = features_t.to(device)
    positions = positions.to(device)
    targets = targets.to(device)

    model = Quantum2DGridModel(
        feature_dim=features_t.shape[1],
        positional_dim=CIFAR_POSITIONAL_DIM,
    ).to(device)
    decoder = PatchDecoder2D(
        positional_dim=CIFAR_POSITIONAL_DIM,
        patch_size=CIFAR_PATCH_SIZE,
    ).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=CIFAR_LR
    )

    order_rows = []
    previous_order = None
    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        recon_loss = torch.mean((output - targets) ** 2)
        energy_loss = torch.mean(energies)
        loss = recon_loss + 0.01 * energy_loss
        loss.backward()
        optimizer.step()
        order_row = quantum_order_summary(
            observables,
            energies,
            step,
            loss.item(),
            recon_loss.item(),
            previous_order,
        )
        previous_order = order_row["mean_order_parameter"]
        order_rows.append(order_row)
        if step % 50 == 0 or step == epochs - 1:
            print(f"  HVK2D Step {step:>4d}: loss={loss.item():.6f} recon={recon_loss.item():.6f}")

    model.eval()
    decoder.eval()
    with torch.no_grad():
        obs, en = model(features_t, positions)
        pred = decoder(obs, positions).cpu().numpy()

    reconstruction = blend_seams(
        stictch_patches(pred, image_size=CIFAR_IMAGE_SIZE, patch_size=CIFAR_PATCH_SIZE),
        patch_size=CIFAR_PATCH_SIZE,
    )
    return reconstruction, order_rows


def run(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    rows = []
    paths = image_paths(args.dataset_dir, args.count)
    for img_path in paths:
        print(f"\nHVK2D processing: {img_path.name}")
        image = load_grayscale_image(img_path)
        reconstruction, order_rows = train_hvk2d(image, device, args.epochs)
        metrics = compute_metrics(reconstruction, image)
        rows.append({
            "model": "HVK2D",
            "image": img_path.name,
            "image_size": CIFAR_IMAGE_SIZE,
            "patch_size": CIFAR_PATCH_SIZE,
            "epochs": args.epochs,
            **metrics,
        })
        output_dir = Path(__file__).resolve().parent / "outputs"
        stem = img_path.stem
        write_csv(output_dir / f"{stem}_hvk2d_order_parameters.csv", order_rows)
        save_order_parameter_plot(
            order_rows,
            output_dir / f"{stem}_hvk2d_order_parameters.png",
            f"HVK2D CIFAR order parameters: {img_path.name}",
        )
        save_reconstruction_comparison(
            image,
            reconstruction,
            output_dir / f"{stem}_hvk2d_reconstruction.png",
            f"HVK2D CIFAR reconstruction: {img_path.name}",
        )
        print(f"  MSE={metrics['mse']:.6f} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HVK2D on CIFAR-10 at 32×32.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=CIFAR_EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything()
    rows = run(args)
    output_dir = Path(__file__).resolve().parent / "outputs"
    write_csv(output_dir / "hvk2d_cifar32_metrics.csv", rows)
    print(f"\nResults saved to {output_dir / 'hvk2d_cifar32_metrics.csv'}")
