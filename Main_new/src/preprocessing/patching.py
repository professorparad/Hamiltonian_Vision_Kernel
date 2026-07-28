import numpy as np


def extract_patches(image: np.ndarray, patch_size: int = 64, stride: int | None = None):
    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    if stride is None:
        stride = patch_size
    if stride <= 0:
        raise ValueError("stride must be positive")

    height, width = image.shape
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size must fit within image dimensions")
    if stride == patch_size and (height % patch_size != 0 or width % patch_size != 0):
        raise ValueError("image dimensions must be divisible by patch_size")

    patches = []
    positions = []
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patches.append(image[i : i + patch_size, j : j + patch_size])
            positions.append([i / height, j / width])

    patches = np.array(patches, dtype=np.float32)
    positions = np.array(positions, dtype=np.float32)
    return patches, positions
