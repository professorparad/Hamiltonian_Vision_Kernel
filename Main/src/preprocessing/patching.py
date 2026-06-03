import numpy as np


def extract_patches(image: np.ndarray, patch_size: int = 64):
    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")

    height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image dimensions must be divisible by patch_size")

    patches = []
    positions = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patches.append(image[i:i + patch_size, j:j + patch_size])
            positions.append([i / height, j / width])

    patches = np.array(patches, dtype=np.float32)
    positions = np.array(positions, dtype=np.float32)
    return patches, positions
