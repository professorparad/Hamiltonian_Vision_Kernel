import numpy as np
import torch


def sinusodial_encoding(
    positions: np.ndarray,
    D_encoded_vector_space: int = 8,
    base: float = 10000.0,
    device=None,
) -> torch.Tensor:
    if D_encoded_vector_space % 4 != 0:
        raise ValueError("D_encoded_vector_space must be divisible by 4.")

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")

    x = positions[:, 0]
    y = positions[:, 1]
    n_positions = positions.shape[0]
    n_bands = D_encoded_vector_space // 4
    encoding = np.zeros((n_positions, D_encoded_vector_space), dtype=np.float32)
    frequencies = np.exp(-np.arange(n_bands, dtype=np.float32) * np.log(base) / n_bands)

    for i, freq in enumerate(frequencies):
        angle_x = 2.0 * np.pi * x * freq
        angle_y = 2.0 * np.pi * y * freq
        encoding[:, 4 * i + 0] = np.sin(angle_x)
        encoding[:, 4 * i + 1] = np.cos(angle_x)
        encoding[:, 4 * i + 2] = np.sin(angle_y)
        encoding[:, 4 * i + 3] = np.cos(angle_y)

    return torch.tensor(encoding, dtype=torch.float32, device=device)


def sinusoidal_positional_encoding(
    positions: np.ndarray,
    d_model: int = 8,
    base: float = 10000.0,
    device=None,
) -> torch.Tensor:
    return sinusodial_encoding(
        positions,
        D_encoded_vector_space=d_model,
        base=base,
        device=device,
    )
