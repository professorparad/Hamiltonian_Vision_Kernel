from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = REPO_ROOT / "Main" / "data" / "monalisa.jpg"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Baselines" / "outputs"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_grayscale_image(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_mask(path: Path, image_size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)


def otsu_pseudo_mask(image: np.ndarray) -> np.ndarray:
    uint_image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(uint_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (mask.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)


def save_grayscale(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image_uint8)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")
    height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    patches = []
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patches.append(image[row : row + patch_size, col : col + patch_size])
    return np.asarray(patches, dtype=np.float32)


def stitch_patches(patches: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    grid = image_size // patch_size
    image = np.zeros((image_size, image_size), dtype=np.float32)
    index = 0
    for row in range(grid):
        for col in range(grid):
            r0 = row * patch_size
            c0 = col * patch_size
            image[r0 : r0 + patch_size, c0 : c0 + patch_size] = patches[index]
            index += 1
    return image


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction.astype(np.float32) - target.astype(np.float32)) ** 2))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    value = mse(prediction, target)
    if value <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(value)))


def simple_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(x.var())
    var_y = float(y.var())
    cov_xy = float(((x - mu_x) * (y - mu_y)).mean())
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator)


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    truth = target.astype(bool)

    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())

    eps = 1e-8
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }
