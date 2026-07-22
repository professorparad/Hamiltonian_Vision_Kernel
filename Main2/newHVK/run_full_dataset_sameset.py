"""Same-set (not held-out) HVK1D/HVK2D reconstruction results across CIFAR-10
and five further real datasets, at a larger sample size than the paper's
original five-image CIFAR-10/Monalisa same-set table. Each image gets its own
freshly trained HVK2D model (non-overlapping 8x8 patches, 16 patches/image),
matching the per-image protocol already established for the hardware pilot.
No held-out split: this reports fit quality, the same category of evidence as
the paper's existing same-set CIFAR-10 table (Table VIII).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "Main"
BENCH_DIR = ROOT / "Baselines" / "cifar10_comparisons"
for p in (MAIN_DIR, BENCH_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.training.training import resolve_device
from Main2.src.model import Quantum2DGridModel, PatchDecoder as PatchDecoder2D
from src.tensornetworks.mps_features import extract_mps_features

PATCH_SIZE = 8
PATCH_STRIDE = 8
N_SITES = 6
POSITIONAL_DIM = 4
IMAGE_SIZE = 32
LR = 0.004

WORKSPACE = ROOT / "Main2" / "newHVK"
OUTPUT_DIR = WORKSPACE / "results" / "full_dataset_sameset"


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    height, width = image.shape
    patches, positions = [], []
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patches.append(image[i : i + patch_size, j : j + patch_size])
            positions.append([i / height, j / width])
    return np.array(patches, dtype=np.float32), np.array(positions, dtype=np.float32)


def resize_to_32(image: np.ndarray) -> np.ndarray:
    import cv2

    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_cifar_images(n: int) -> list[np.ndarray]:
    import cv2

    image_dir = BENCH_DIR / "datasets" / "images"
    paths = sorted(image_dir.glob("*.png"))
    if len(paths) < n:
        # Fall back to the raw CIFAR-10 tarball (already cached by download_cifar32.py)
        # for additional images beyond the small curated visual subset.
        cache_dir = ROOT / "Baselines" / "datasets" / "_cache"
        archive = cache_dir / "cifar-10-python.tar.gz"
        if archive.exists():
            import pickle
            import tarfile

            with tarfile.open(archive, "r:gz") as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("data_batch_1"))
                with tar.extractfile(member) as f:
                    batch = pickle.load(f, encoding="latin1")
            extra = batch["data"].reshape(-1, 3, 32, 32).mean(axis=1) / 255.0
            images = [img.astype(np.float32) for img in extra[:n]]
            return images
    images = []
    for p in paths[:n]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        images.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))
    return images


def load_torchvision_images(name: str, n: int) -> list[np.ndarray]:
    from torchvision import datasets

    root = WORKSPACE / "datasets"
    if name == "mnist":
        dataset = datasets.MNIST(root=str(root), train=True, download=True)
    elif name == "fashion-mnist":
        dataset = datasets.FashionMNIST(root=str(root), train=True, download=True)
    else:
        raise ValueError(name)
    data = dataset.data.numpy() if hasattr(dataset.data, "numpy") else np.asarray(dataset.data)
    return [resize_to_32(data[i]) for i in range(n)]


def load_medmnist_images(name: str, n: int) -> list[np.ndarray]:
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    dataset_class = getattr(medmnist, info["python_class"])
    root = WORKSPACE / "datasets"
    dataset = dataset_class(split="train", root=str(root), download=True, as_rgb=False, size=28)
    imgs = np.asarray(dataset.imgs)
    if imgs.ndim == 4:
        imgs = imgs.mean(axis=-1)
    return [resize_to_32(imgs[i]) for i in range(n)]


DATASET_LOADERS = {
    "cifar10": lambda n: load_cifar_images(n),
    "mnist": lambda n: load_torchvision_images("mnist", n),
    "fashion-mnist": lambda n: load_torchvision_images("fashion-mnist", n),
    "pathmnist": lambda n: load_medmnist_images("pathmnist", n),
    "bloodmnist": lambda n: load_medmnist_images("bloodmnist", n),
    "pneumoniamnist": lambda n: load_medmnist_images("pneumoniamnist", n),
}


def psnr_ssim(pred: np.ndarray, target: np.ndarray) -> dict:
    mse = float(np.mean((pred - target) ** 2))
    psnr = 20 * np.log10(1.0 / np.sqrt(max(mse, 1e-12)))
    # lightweight SSIM (global, single-scale) to avoid an extra dependency
    mu_p, mu_t = pred.mean(), target.mean()
    var_p, var_t = pred.var(), target.var()
    cov = ((pred - mu_p) * (target - mu_t)).mean()
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2))
    return {"mse": mse, "psnr": psnr, "ssim": float(ssim)}


def train_one(image: np.ndarray, device: torch.device, epochs: int) -> dict:
    patches, raw_positions = extract_patches(image, PATCH_SIZE, PATCH_STRIDE)
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in (patches + 1e-4)])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = Quantum2DGridModel(feature_dim=features_t.shape[1], positional_dim=POSITIONAL_DIM).to(device)
    decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()

    model.eval()
    decoder.eval()
    with torch.no_grad():
        obs, en = model(features_t, positions)
        pred = decoder(obs, positions).cpu().numpy()[:, 0]

    pred_full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for idx, (i, j) in enumerate(raw_positions):
        ii, jj = int(round(i * IMAGE_SIZE)), int(round(j * IMAGE_SIZE))
        pred_full[ii : ii + PATCH_SIZE, jj : jj + PATCH_SIZE] = pred[idx]
    return psnr_ssim(pred_full, image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-per-dataset", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LOADERS.keys()))
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary = []
    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        images = DATASET_LOADERS[name](args.images_per_dataset)
        rows = []
        for idx, image in enumerate(images):
            if args.time_budget_minutes and (time.time() - start) / 60 > args.time_budget_minutes:
                print(f"  time budget of {args.time_budget_minutes} min reached, stopping early.", flush=True)
                break
            try:
                metrics = train_one(image, device, args.epochs)
            except Exception as exc:
                print(f"  image {idx}: SKIPPED due to error: {exc}", flush=True)
                continue
            rows.append(metrics)
            elapsed = (time.time() - start) / 60
            print(f"  image {idx}: psnr={metrics['psnr']:.2f} ssim={metrics['ssim']:.4f} (elapsed {elapsed:.1f} min)", flush=True)
        if not rows:
            continue
        psnrs = [r["psnr"] for r in rows]
        ssims = [r["ssim"] for r in rows]
        summary.append(
            {
                "dataset": name,
                "n_images": len(rows),
                "mean_psnr": float(np.mean(psnrs)),
                "std_psnr": float(np.std(psnrs)),
                "mean_ssim": float(np.mean(ssims)),
                "std_ssim": float(np.std(ssims)),
            }
        )
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        if args.time_budget_minutes and (time.time() - start) / 60 > args.time_budget_minutes:
            break

    print("\n=== Summary ===")
    for row in summary:
        print(f"{row['dataset']}: n={row['n_images']} PSNR={row['mean_psnr']:.2f}+/-{row['std_psnr']:.2f} SSIM={row['mean_ssim']:.4f}+/-{row['std_ssim']:.4f}")
    print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
