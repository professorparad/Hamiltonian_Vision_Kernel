import matplotlib.pyplot as plt
import numpy as np


def plot_reconstructions(
    original: np.ndarray,
    quantum_reconstruction: np.ndarray,
    mps_baseline: np.ndarray,
    random_latent: np.ndarray,
    zero_latent: np.ndarray,
):

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    panels = [
        ("Original", original),
        ("Quantum Reconstruction", quantum_reconstruction),
        ("MPS Baseline", mps_baseline),
        ("Random Latent", random_latent),
        ("Zero Latent", zero_latent),
    ]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1)

        ax.set_title(title, fontsize=10)

        ax.axis("off")

    plt.suptitle("Hamiltonian Vision Kernel", fontsize=14, y=1.02)

    plt.tight_layout()

    plt.show()
