"""Rebuild of the zero-shot / multi-image adaptation claim in
supplementary_study.tex (Table tab:generalization_controls, Section
"Adaptation Beyond Single-Image Optimization"): "a model optimized on
Monalisa alone does not transfer zero-shot to a structurally different
image ... extending training to include a second image recovers"
high fidelity on it.

No script backing that table's specific 7.78/28.31 dB numbers was ever found
in the repo (see TODO/results-core-map.md, row R2/A3) despite an exhaustive
search, and the original choice of "second image" was undocumented and is
unrecoverable. This script does not attempt to reproduce those exact numbers
-- it reruns the described protocol fresh, using the one image that has sat
alongside monalisa.jpg in every data directory in this repo
(Main/data/handofgod_micheal_angelo.jpg) without ever being used by any
committed script, making it the natural candidate for the undocumented
"second, structurally different image."

Protocol, matching the same-set per-image training already used for the
paper's Table II (Main2/newHVK/run_full_dataset_sameset.py::train_one --
same Quantum2DGridModel + PatchDecoder2D architecture, same 8x8 patches,
same per-image feature normalization):

  1. Train on Monalisa alone for `--epochs` steps.
  2. Zero-shot: evaluate the frozen Monalisa-trained model directly on
     Hand of God's patches (no further training) -- this is what "zero-shot
     transfer" means: the model has never seen this image's patches or
     gradients.
  3. Multi-image: continue training the same model+decoder, now on the
     concatenation of both images' patches, for another `--epochs` steps,
     then evaluate on Hand of God again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "Main"
for p in (MAIN_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.preprocessing.positional_encoding import sinusoidal_positional_encoding  # noqa: E402
from src.training.training import resolve_device  # noqa: E402
from Main2.src.model import PatchDecoder as PatchDecoder2D  # noqa: E402
from Main2.src.model import Quantum2DGridModel  # noqa: E402
from Main2.newHVK.run_full_dataset_sameset import extract_patches, psnr_ssim  # noqa: E402
from src.tensornetworks.mps_features import extract_mps_features  # noqa: E402

PATCH_SIZE = 8
PATCH_STRIDE = 8
N_SITES = 6
POSITIONAL_DIM = 4
IMAGE_SIZE = 32
LR = 0.004

WORKSPACE = ROOT / "Main2" / "newHVK"
OUTPUT_DIR = WORKSPACE / "results" / "zero_shot_generalization"
MONALISA_PATH = MAIN_DIR / "data" / "monalisa.jpg"
SECOND_IMAGE_PATH = MAIN_DIR / "data" / "handofgod_micheal_angelo.jpg"


def load_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def prepare(image: np.ndarray, device: torch.device):
    patches, raw_positions = extract_patches(image, PATCH_SIZE, PATCH_STRIDE)
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in (patches + 1e-4)])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    return (
        features_t.to(device),
        positions.to(device),
        targets.to(device),
        raw_positions,
    )


def reconstruct(model, decoder, features_t, positions, raw_positions, image_shape):
    model.eval()
    decoder.eval()
    with torch.no_grad():
        obs, _ = model(features_t, positions)
        pred = decoder(obs, positions).cpu().numpy()[:, 0]
    pred_full = np.zeros(image_shape, dtype=np.float32)
    for idx, (i, j) in enumerate(raw_positions):
        ii, jj = int(round(i * IMAGE_SIZE)), int(round(j * IMAGE_SIZE))
        pred_full[ii : ii + PATCH_SIZE, jj : jj + PATCH_SIZE] = pred[idx]
    return pred_full


def train_step(model, decoder, optimizer, features_t, positions, targets, epochs: int):
    for _ in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monalisa = load_image(MONALISA_PATH)
    second = load_image(SECOND_IMAGE_PATH)

    ml_features, ml_positions, ml_targets, ml_raw_pos = prepare(monalisa, device)
    sc_features, sc_positions, sc_targets, sc_raw_pos = prepare(second, device)

    model = Quantum2DGridModel(feature_dim=ml_features.shape[1], positional_dim=POSITIONAL_DIM).to(device)
    decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    print("Step 1: training on Monalisa alone...", flush=True)
    train_step(model, decoder, optimizer, ml_features, ml_positions, ml_targets, args.epochs)
    monalisa_pred = reconstruct(model, decoder, ml_features, ml_positions, ml_raw_pos, monalisa.shape)
    monalisa_metrics = psnr_ssim(monalisa_pred, monalisa)
    print(f"  Monalisa (same-image, sanity check): PSNR={monalisa_metrics['psnr']:.2f} SSIM={monalisa_metrics['ssim']:.4f}", flush=True)

    print("Step 2: zero-shot evaluation on the second image (no further training)...", flush=True)
    zero_shot_pred = reconstruct(model, decoder, sc_features, sc_positions, sc_raw_pos, second.shape)
    zero_shot_metrics = psnr_ssim(zero_shot_pred, second)
    print(f"  Second image, zero-shot: PSNR={zero_shot_metrics['psnr']:.2f} SSIM={zero_shot_metrics['ssim']:.4f}", flush=True)

    print("Step 3: extending training to include the second image...", flush=True)
    combined_features = torch.cat([ml_features, sc_features], dim=0)
    combined_positions = torch.cat([ml_positions, sc_positions], dim=0)
    combined_targets = torch.cat([ml_targets, sc_targets], dim=0)
    train_step(model, decoder, optimizer, combined_features, combined_positions, combined_targets, args.epochs)
    multi_image_pred = reconstruct(model, decoder, sc_features, sc_positions, sc_raw_pos, second.shape)
    multi_image_metrics = psnr_ssim(multi_image_pred, second)
    print(f"  Second image, multi-image training: PSNR={multi_image_metrics['psnr']:.2f} SSIM={multi_image_metrics['ssim']:.4f}", flush=True)

    result = {
        "protocol": (
            "Fresh rerun, 2026-07-29 -- see script docstring. Second image is "
            "Main/data/handofgod_micheal_angelo.jpg, chosen because it is the "
            "only other image bundled alongside monalisa.jpg in this repo's "
            "data directories and was otherwise unused by any committed "
            "script. This does NOT reproduce the original manuscript's "
            "7.78/28.31 dB numbers -- no source for that original run was "
            "found; these are new, freshly-verified numbers under a "
            "documented protocol."
        ),
        "epochs_per_stage": args.epochs,
        "seed": args.seed,
        "monalisa_same_image_sanity_check": monalisa_metrics,
        "second_image_zero_shot": zero_shot_metrics,
        "second_image_multi_image_training": multi_image_metrics,
    }
    output_path = OUTPUT_DIR / "summary.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
