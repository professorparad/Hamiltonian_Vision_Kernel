import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_map(
    entropy_features: np.ndarray, grid_size: tuple = (4, 4), cmap: str = "inferno"
):
    """
    Visualize average patch entanglement entropy.

    Parameters
    ----------
    entropy_features : np.ndarray
        Shape (N, n_entropy_features)

    grid_size : tuple
        Patch grid dimensions.

    cmap : str
        Matplotlib colormap.
    """

    if entropy_features.ndim != 2:
        raise ValueError("entropy_features must have shape (N, M)")

    patch_entropy = entropy_features.mean(axis=1)

    entropy_map = patch_entropy.reshape(grid_size)

    plt.figure(figsize=(6, 6))

    image = plt.imshow(entropy_map, cmap=cmap)

    plt.colorbar(image, label="Average Entanglement Entropy")

    plt.title("HVK Entanglement Entropy Map")

    plt.axis("off")

    plt.tight_layout()

    plt.show()
