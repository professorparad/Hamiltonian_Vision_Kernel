"""Workstream 3 (real-circuit confirmation): D4 equivariance, actual trained
quantum circuit rather than the analytic surrogate.

Reuses an already-trained HVK2D checkpoint (no retraining) from
Baselines/cifar10_comparisons/hvk2d/hardware_checkpoints/ -- the same
checkpoints used for the real IBM hardware pilot. For each of a handful of
CIFAR-10 images: apply all 8 D4 transforms to the full image, run the
TRAINED model's real PennyLane quantum-circuit forward pass (exact
statevector simulation of the actual trained weights, not the closed-form
surrogate formula) on each transform's patches, and measure both the
non-pooled feature-equivariance error and the D4-pooled (Reynolds-averaged)
equivariance error -- confirming whether the surrogate's near-machine-
precision pooled result and large non-pooled gap actually hold for the real
trained circuit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")
BENCH_ROOT = REPO_ROOT / "Baselines" / "cifar10_comparisons"
MAIN_DIR = REPO_ROOT / "Main"
for p in (BENCH_ROOT, MAIN_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import load_grayscale_image
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from Main2.src.model import PatchDecoder as PatchDecoder2D
from Main2.src.model import Quantum2DGridModel

CHECKPOINT_ROOT = BENCH_ROOT / "hvk2d" / "hardware_checkpoints"
CIFAR_DIR = BENCH_ROOT / "datasets" / "images"
PATCH_SIZE = 8
N_SITES = 6
POSITIONAL_DIM = 4
IMAGE_SIZE = 32

OUT_DIR = REPO_ROOT / "Main2" / "newHVK" / "results" / "d4_symmetry_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def d4_transforms(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": image.copy(),
        "rot90": np.rot90(image, 1),
        "rot180": np.rot90(image, 2),
        "rot270": np.rot90(image, 3),
        "flip_h": np.fliplr(image),
        "flip_v": np.flipud(image),
        "transpose": image.T,
        "anti_transpose": np.fliplr(np.flipud(image)).T,
    }


def inverse_transform_grid(grid: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return grid.copy()
    if transform == "rot90":
        return np.rot90(grid, -1, axes=(0, 1))
    if transform == "rot180":
        return np.rot90(grid, 2, axes=(0, 1))
    if transform == "rot270":
        return np.rot90(grid, 1, axes=(0, 1))
    if transform == "flip_h":
        return np.fliplr(grid)
    if transform == "flip_v":
        return np.flipud(grid)
    if transform == "transpose":
        return np.swapaxes(grid, 0, 1)
    if transform == "anti_transpose":
        return np.flipud(np.fliplr(np.swapaxes(grid, 0, 1)))
    raise ValueError(transform)


def load_checkpoint(stem: str):
    ckpt_dir = CHECKPOINT_ROOT / stem
    sample_patch = np.load(ckpt_dir / "patches.npy")[0]
    feature_dim = extract_mps_features(sample_patch + 1e-4, n_sites=N_SITES, bond_dim=4).shape[0]
    model = Quantum2DGridModel(feature_dim=feature_dim, positional_dim=POSITIONAL_DIM)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.eval()
    decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE)
    decoder.load_state_dict(torch.load(ckpt_dir / "decoder.pt", map_location="cpu"))
    decoder.eval()
    return model, decoder


def real_circuit_grid(image: np.ndarray, model, decoder) -> tuple[np.ndarray, np.ndarray]:
    """Returns (observable_grid [4,4,obs_dim], reconstruction_grid [4,4,64])
    from the actual trained PennyLane quantum-circuit forward pass."""
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_SIZE)
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions_t = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)

    with torch.no_grad():
        observables, _ = model(features_t, positions_t)
        pred = decoder(observables, positions_t).numpy()

    obs_np = observables.numpy()
    grid_size = int(np.sqrt(len(patches)))
    obs_grid = obs_np.reshape(grid_size, grid_size, -1)
    pred_grid = pred.reshape(grid_size, grid_size, -1)
    return obs_grid, pred_grid


def d4_pooled_grid(image: np.ndarray, model, decoder) -> tuple[np.ndarray, np.ndarray]:
    obs_aligned, pred_aligned = [], []
    for name, transformed in d4_transforms(image).items():
        obs_grid, pred_grid = real_circuit_grid(transformed, model, decoder)
        obs_aligned.append(inverse_transform_grid(obs_grid, name))
        pred_aligned.append(inverse_transform_grid(pred_grid, name))
    return np.mean(obs_aligned, axis=0), np.mean(pred_aligned, axis=0)


def equivariance_and_consistency(image: np.ndarray, model, decoder, pooled: bool) -> dict:
    if pooled:
        base_obs, base_pred = d4_pooled_grid(image, model, decoder)
    else:
        base_obs, base_pred = real_circuit_grid(image, model, decoder)
    denom = float(np.linalg.norm(base_obs) + 1e-8)

    equiv_errors = []
    canon_preds = [base_pred]
    for name, transformed in d4_transforms(image).items():
        if name == "identity":
            continue
        if pooled:
            t_obs, t_pred = d4_pooled_grid(transformed, model, decoder)
        else:
            t_obs, t_pred = real_circuit_grid(transformed, model, decoder)
        t_obs_canon = inverse_transform_grid(t_obs, name)
        t_pred_canon = inverse_transform_grid(t_pred, name)
        equiv_errors.append(float(np.linalg.norm(base_obs - t_obs_canon) / denom))
        canon_preds.append(t_pred_canon)

    canon_preds = np.stack(canon_preds, axis=0)
    mean_pred = canon_preds.mean(axis=0)
    consistency_mse = float(np.mean([(p - mean_pred) ** 2 for p in canon_preds]))
    return {
        "mean_equivariance_error": float(np.mean(equiv_errors)),
        "max_equivariance_error": float(np.max(equiv_errors)),
        "output_consistency_mse": consistency_mse,
    }


def main():
    stems = sorted(p.name for p in CHECKPOINT_ROOT.iterdir() if p.is_dir())
    results = []
    for stem in stems:
        print(f"\n=== {stem} ===", flush=True)
        model, decoder = load_checkpoint(stem)
        image_path = sorted(CIFAR_DIR.glob(f"{stem}.png"))[0]
        image = load_grayscale_image(image_path)

        r_nonpooled = equivariance_and_consistency(image, model, decoder, pooled=False)
        print("  non-pooled (real trained circuit):", json.dumps(r_nonpooled), flush=True)
        r_pooled = equivariance_and_consistency(image, model, decoder, pooled=True)
        print("  D4-pooled  (real trained circuit):", json.dumps(r_pooled), flush=True)

        results.append({"image": stem, "mode": "non-pooled-real-circuit", **r_nonpooled})
        results.append({"image": stem, "mode": "D4-pooled-real-circuit", **r_pooled})
        (OUT_DIR / "d4_real_circuit_confirmation.json").write_text(json.dumps(results, indent=2))

    print("\nDone. Saved to", OUT_DIR / "d4_real_circuit_confirmation.json")


if __name__ == "__main__":
    main()
