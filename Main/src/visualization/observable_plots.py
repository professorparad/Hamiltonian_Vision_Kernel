import matplotlib.pyplot as plt
import numpy as np


def plot_observables(observables: np.ndarray, n_qubits: int = 6):

    z_obs = observables[:, :n_qubits]

    x_obs = observables[:, n_qubits : 2 * n_qubits]

    zz_corr = observables[:, 2 * n_qubits : 2 * n_qubits + (n_qubits - 1)]

    xx_corr = observables[
        :, 2 * n_qubits + (n_qubits - 1) : 2 * n_qubits + 2 * (n_qubits - 1)
    ]

    yy_corr = observables[:, 2 * n_qubits + 2 * (n_qubits - 1) :]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    panels = [
        ("Local Z", z_obs),
        ("Local X", x_obs),
        ("ZZ Correlations", zz_corr),
        ("XX Correlations", xx_corr),
        ("YY Correlations", yy_corr),
    ]

    for ax, (title, data) in zip(axes, panels):
        image = ax.imshow(data, aspect="auto", cmap="coolwarm")

        ax.set_title(title)

        plt.colorbar(image, ax=ax, fraction=0.046)

    plt.suptitle("Quantum Observable Maps", fontsize=14)

    plt.tight_layout()

    plt.show()
