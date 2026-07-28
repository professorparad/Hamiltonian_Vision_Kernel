"""Dataset-level generalization study: a single shared HVK2D model trained
via gradient descent across many CIFAR-10 images, with a genuine train/held-out
image split, evaluated against a parameter-matched classical control.

Every reconstruction result reported elsewhere in this project (Sections
IV-A--IV-B of the main paper) trains a *fresh* model per target image: no
held-out split, so those numbers measure per-image fitting, not
generalization. The existing "held-out" comparison in the companion
supplementary study (run_newhvk_suite.py:run_real_cifar_holdout) is a ridge
regression on a hand-crafted closed-form feature transform (pairwise
products + sine harmonics of classical MPS features) fit on only 6 training
images and evaluated on 4 held-out images -- it never trains the actual
gradient-trained VQC (Quantum2DGridModel) at all.

This script closes that gap: ONE Quantum2DGridModel + PatchDecoder2D (the
paper's actual HVK2D architecture, hvk.hvk2d.model) is trained jointly, via
full-batch Adam, across all patches from N_TRAIN images, then evaluated
(no gradient, no retraining) on N_HELDOUT images the model never saw during
training. A parameter-matched classical control (ClassicalLinearControl:
a single linear map from the same features+positions to the same
observable dimensionality, feeding the identical decoder, energy=0) is
trained identically on the same split, to isolate what the learned quantum
circuit contributes beyond a linear feature map under a shared decoder.

Feature normalization statistics are computed from the training split only
and applied to the held-out split (no test-set leakage). Reported per-seed
so the run can be interrupted and resumed; each seed reshuffles the
train/held-out image split (matching the existing single-image held-out
study's convention of a fresh seeded permutation per seed).

Compute-budget note: the quantum circuit is a per-patch PennyLane
simulation executed in a Python loop (not vectorized), so cost scales
linearly with image count. EPOCHS is set to fit a multi-hour, not
multi-day, budget; this is a training-budget constraint, not a claim of
full convergence, and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "Main"
BENCH_DIR = ROOT / "Baselines" / "cifar10_comparisons"
PY_LIB = ROOT / "python_library" / "src"
for p in (MAIN_DIR, BENCH_DIR, ROOT, PY_LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import compute_metrics, stitch_patches  # noqa: E402
from hvk.hvk2d.model import OBS_DIM, PatchDecoder2D, Quantum2DGridModel  # noqa: E402
from src.preprocessing.patching import extract_patches  # noqa: E402
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding  # noqa: E402
from src.tensornetworks.mps_features import extract_mps_features  # noqa: E402
from src.training.training import resolve_device  # noqa: E402

IMAGE_SIZE = 32
PATCH_SIZE = 8
N_SITES = 6
POSITIONAL_DIM = 4
BOND_DIM = 4

N_TRAIN = 150
N_HELDOUT = 50
SEEDS = [0, 1, 2]
EPOCHS = 20  # full-batch epochs over 2400 train patches; ~137ms/patch measured -> ~5.5hr/3 seeds
LR = 0.004

CIFAR_DIR = BENCH_DIR / "datasets" / "images"
OUTPUT_DIR = ROOT / "Main2" / "newHVK" / "results" / "dataset_level_generalization"


class ClassicalLinearControl(nn.Module):
    """Parameter-matched control: same decoder input width, no quantum circuit."""

    def __init__(self, feature_dim: int, positional_dim: int, obs_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim + positional_dim, obs_dim)

    def forward(self, features: torch.Tensor, positions: torch.Tensor):
        combined = torch.cat([features, positions], dim=-1)
        observables = self.linear(combined)
        energies = torch.zeros(observables.shape[0], device=observables.device)
        return observables, energies


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32) / 255.0
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def build_patch_table(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int, int]]]:
    """Returns (features, positions_raw, targets, meta) with meta[i] = (image_name, image_idx, patch_idx)."""
    all_features, all_positions, all_targets, meta = [], [], [], []
    for img_idx, path in enumerate(paths):
        image = load_image(path)
        patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_SIZE)
        safe_patches = patches + 1e-4
        features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=BOND_DIM) for p in safe_patches])
        all_features.append(features)
        all_positions.append(raw_positions)
        all_targets.append(patches)
        for patch_idx in range(len(patches)):
            meta.append((path.stem, img_idx, patch_idx))
    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_positions, axis=0),
        np.concatenate(all_targets, axis=0),
        meta,
    )


def train_shared_model(
    model: nn.Module,
    decoder: PatchDecoder2D,
    features_t: torch.Tensor,
    positions_t: torch.Tensor,
    targets_t: torch.Tensor,
    epochs: int,
    lr: float,
) -> list[float]:
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
    loss_history = []
    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions_t)
        output = decoder(observables, positions_t)
        recon_loss = torch.mean((output - targets_t) ** 2)
        loss = recon_loss + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        loss_history.append(float(recon_loss.item()))
        if step % 5 == 0 or step == epochs - 1:
            print(f"    epoch {step:>3d}: recon_mse={recon_loss.item():.6f}", flush=True)
    return loss_history


def evaluate_heldout(
    model: nn.Module,
    decoder: PatchDecoder2D,
    features_t: torch.Tensor,
    positions_t: torch.Tensor,
    targets: np.ndarray,
    meta: list[tuple[str, int, int]],
) -> list[dict]:
    model.eval()
    decoder.eval()
    with torch.no_grad():
        observables, _ = model(features_t, positions_t)
        pred = decoder(observables, positions_t).cpu().numpy()

    by_image: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for (name, img_idx, patch_idx), pred_patch, target_patch in zip(meta, pred, targets):
        by_image.setdefault(img_idx, []).append((patch_idx, pred_patch, target_patch))

    rows = []
    name_by_idx = {img_idx: name for name, img_idx, _ in meta}
    grid = IMAGE_SIZE // PATCH_SIZE
    for img_idx, entries in sorted(by_image.items()):
        entries.sort(key=lambda e: e[0])
        pred_patches = np.array([e[1].squeeze() for e in entries])
        target_patches = np.array([e[2].squeeze() for e in entries])
        pred_image = stitch_patches(pred_patches, image_size=grid * PATCH_SIZE, patch_size=PATCH_SIZE)
        target_image = stitch_patches(target_patches, image_size=grid * PATCH_SIZE, patch_size=PATCH_SIZE)
        metrics = compute_metrics(pred_image, target_image)
        rows.append({"image": name_by_idx[img_idx], **metrics})
    return rows


def run_seed(seed: int, all_paths: list[Path], epochs: int, device: torch.device) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_paths))
    train_paths = [all_paths[i] for i in order[:N_TRAIN]]
    heldout_paths = [all_paths[i] for i in order[N_TRAIN : N_TRAIN + N_HELDOUT]]

    print(f"  building patch table: {len(train_paths)} train images...", flush=True)
    train_features, train_positions_raw, train_targets, _ = build_patch_table(train_paths)
    print(f"  building patch table: {len(heldout_paths)} held-out images...", flush=True)
    heldout_features, heldout_positions_raw, heldout_targets, heldout_meta = build_patch_table(heldout_paths)

    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0) + 1e-8

    def normalize(features: np.ndarray) -> torch.Tensor:
        return torch.tensor((features - feat_mean) / feat_std, dtype=torch.float32).to(device)

    train_features_t = normalize(train_features)
    heldout_features_t = normalize(heldout_features)
    train_positions_t = sinusoidal_positional_encoding(train_positions_raw, d_model=POSITIONAL_DIM).to(device)
    heldout_positions_t = sinusoidal_positional_encoding(heldout_positions_raw, d_model=POSITIONAL_DIM).to(device)
    train_targets_t = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1).to(device)

    result: dict = {"seed": seed, "train_images": [p.stem for p in train_paths], "heldout_images": [p.stem for p in heldout_paths]}

    for model_name in ("hvk2d_quantum", "classical_linear_control"):
        print(f"  training {model_name} (seed={seed}, {len(train_paths)} images, {epochs} epochs)...", flush=True)
        set_seed(seed)
        if model_name == "hvk2d_quantum":
            model: nn.Module = Quantum2DGridModel(feature_dim=train_features_t.shape[1], positional_dim=POSITIONAL_DIM).to(device)
        else:
            model = ClassicalLinearControl(feature_dim=train_features_t.shape[1], positional_dim=POSITIONAL_DIM, obs_dim=OBS_DIM).to(device)
        decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)

        t0 = time.time()
        loss_history = train_shared_model(model, decoder, train_features_t, train_positions_t, train_targets_t, epochs, LR)
        train_seconds = time.time() - t0

        heldout_rows = evaluate_heldout(model, decoder, heldout_features_t, heldout_positions_t, heldout_targets, heldout_meta)
        psnrs = [r["psnr"] for r in heldout_rows]
        result[model_name] = {
            "final_train_recon_mse": loss_history[-1],
            "train_seconds": train_seconds,
            "heldout_rows": heldout_rows,
            "heldout_psnr_mean": float(np.mean(psnrs)),
            "heldout_psnr_std": float(np.std(psnrs)),
        }
        print(f"    {model_name}: held-out PSNR = {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB ({train_seconds:.0f}s train)", flush=True)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-heldout", type=int, default=N_HELDOUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    # default cpu: PennyLane's default.qubit simulator doesn't run on GPU, so moving the
    # many tiny per-patch tensors to CUDA only adds transfer overhead (measured ~2.5x slower).
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global N_TRAIN, N_HELDOUT
    N_TRAIN, N_HELDOUT = args.n_train, args.n_heldout
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    all_paths = sorted(CIFAR_DIR.glob("*.png"))
    needed = N_TRAIN + N_HELDOUT
    if len(all_paths) < needed:
        raise FileNotFoundError(f"Need {needed} CIFAR PNGs, found {len(all_paths)} in {CIFAR_DIR}")
    print(f"Found {len(all_paths)} CIFAR images available; using {N_TRAIN} train + {N_HELDOUT} held-out per seed.", flush=True)

    results_path = OUTPUT_DIR / "dataset_level_generalization.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else []
    done_seeds = {r["seed"] for r in all_results}

    for seed in args.seeds:
        if seed in done_seeds:
            print(f"skip (already done): seed={seed}", flush=True)
            continue
        print(f"\n=== seed={seed} ===", flush=True)
        result = run_seed(seed, all_paths, args.epochs, device)
        all_results.append(result)
        results_path.write_text(json.dumps(all_results, indent=2))
        print(f"  saved to {results_path}", flush=True)

    summary = {}
    for model_name in ("hvk2d_quantum", "classical_linear_control"):
        means = [r[model_name]["heldout_psnr_mean"] for r in all_results if model_name in r]
        summary[model_name] = {
            "n_seeds": len(means),
            "psnr_mean_across_seeds": float(np.mean(means)) if means else None,
            "psnr_std_across_seeds": float(np.std(means)) if means else None,
        }
    print("\n=== Summary (mean held-out PSNR across seeds) ===")
    for model_name, s in summary.items():
        print(f"{model_name}: {s['psnr_mean_across_seeds']:.2f} +/- {s['psnr_std_across_seeds']:.2f} dB (n_seeds={s['n_seeds']})")
    (OUTPUT_DIR / "dataset_level_generalization_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
