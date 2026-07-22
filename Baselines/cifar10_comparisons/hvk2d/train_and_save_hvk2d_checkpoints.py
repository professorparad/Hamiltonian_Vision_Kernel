"""Train HVK2D on a few held-out CIFAR-10 images (non-overlapping 8x8 patches,
matching the Monalisa hardware-pilot patch count of 16 per image) and save
model.pt/decoder.pt checkpoints plus per-patch observables, so a hardware
replay script can reuse them without retraining."""
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

from common import DEFAULT_DATASET_DIR, compute_metrics, image_paths, load_grayscale_image
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from Main2.src.model import Quantum2DGridModel, PatchDecoder as PatchDecoder2D
from src.training.training import resolve_device

CIFAR_IMAGE_SIZE = 32
CIFAR_PATCH_SIZE = 8
CIFAR_PATCH_STRIDE = 8  # non-overlapping: 16 patches/image, matching the Monalisa hardware pilot
CIFAR_N_SITES = 6
CIFAR_POSITIONAL_DIM = 4
CIFAR_EPOCHS = 200
CIFAR_LR = 0.004


def train_one(image: np.ndarray, device: torch.device, epochs: int):
    patches, raw_positions = extract_patches(image, patch_size=CIFAR_PATCH_SIZE, stride=CIFAR_PATCH_STRIDE)
    features = np.array([extract_mps_features(p, n_sites=CIFAR_N_SITES, bond_dim=4) for p in patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=CIFAR_POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = Quantum2DGridModel(feature_dim=features_t.shape[1], positional_dim=CIFAR_POSITIONAL_DIM).to(device)
    decoder = PatchDecoder2D(positional_dim=CIFAR_POSITIONAL_DIM, patch_size=CIFAR_PATCH_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=CIFAR_LR)

    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == epochs - 1:
            print(f"    step {step:>4d}: loss={loss.item():.6f}")

    model.eval()
    decoder.eval()
    with torch.no_grad():
        obs, en = model(features_t, positions)
        pred = decoder(obs, positions).cpu().numpy()

    pred_full = np.zeros((CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE), dtype=np.float32)
    for idx, (i, j) in enumerate(raw_positions):
        ii, jj = int(round(i * CIFAR_IMAGE_SIZE)), int(round(j * CIFAR_IMAGE_SIZE))
        pred_full[ii : ii + CIFAR_PATCH_SIZE, jj : jj + CIFAR_PATCH_SIZE] = pred[idx, 0]
    metrics = compute_metrics(pred_full, image)

    return {
        "model_state": model.cpu().state_dict(),
        "decoder_state": decoder.cpu().state_dict(),
        "observables": obs.detach().cpu().numpy(),
        "patches": patches,
        "positions": raw_positions,
        "reconstruction": pred_full,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=CIFAR_EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "hardware_checkpoints")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(args.dataset_dir, args.count)
    for img_path in paths:
        print(f"\nTraining HVK2D on {img_path.name} ({args.epochs} steps)...")
        image = load_grayscale_image(img_path)
        result = train_one(image, device, args.epochs)
        stem = img_path.stem
        out_dir = args.output_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result["model_state"], out_dir / "model.pt")
        torch.save(result["decoder_state"], out_dir / "decoder.pt")
        np.save(out_dir / "observables.npy", result["observables"])
        np.save(out_dir / "patches.npy", result["patches"])
        np.save(out_dir / "positions.npy", result["positions"])
        np.save(out_dir / "reconstruction.npy", result["reconstruction"])
        print(f"  Saved checkpoint to {out_dir}")
        print(f"  Simulator metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
