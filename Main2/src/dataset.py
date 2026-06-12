from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from Main2.src.pathing import add_main_package_to_path

add_main_package_to_path()

from src.preprocessing.image_loader import load_image_grayscale
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features


def build_main2_dataset(
    *,
    image_path: str | Path,
    image_size: int,
    patch_size: int,
    positional_dim: int,
    device: torch.device,
):
    image = load_image_grayscale(str(image_path), size=(image_size, image_size))
    patches, raw_positions = extract_patches(image, patch_size=patch_size)
    features = np.array([extract_mps_features(patch) for patch in patches])
    features = torch.tensor(features, dtype=torch.float32)
    features = (features - features.mean(dim=0)) / (
        features.std(dim=0, unbiased=False) + 1e-8
    )
    positions = sinusoidal_positional_encoding(raw_positions, d_model=positional_dim)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    return {
        "image": image,
        "patches": patches,
        "raw_positions": raw_positions,
        "features": features.to(device),
        "positions": positions.to(device),
        "targets": targets.to(device),
    }
