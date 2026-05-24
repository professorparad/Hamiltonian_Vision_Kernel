import torch
import torch.optim as optim
import numpy as np

from src.preprocessing.image_loader import ( load_image_grayscale
)

from src.preprocessing.patching import (
    extract_patches
)

from src.preprocessing.positional_encoding import (
    sinusodial_encoding
)

from src.tensornetworks.mps_features import (
    extract_mps_features
)

from src.tensornetworks.mps_reconstruction import (
    mps_reconstruct
)

from src.quantum.circuit import (
    observable_dim
)

from src.quantum.quantum_model import (
    QuantumModel
)

from src.decoder.patch_decoder import (
    PatchDecoder
)

from src.reconstruction.patch_stitching import (
    stictch_patches
)

from src.reconstruction.seam_bleading import (
    blend_seams
)

from src.visualization.entropy_maps import (
    plot_entropy_map
)

from src.visualization.observable_plots import (
    plot_observables
)

from src.visualization.training_curve import (
    plot_training_curves
)

from src.visualization.reconstruction_plots import (
    plot_reconstructions
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

img_path = (
    "/home/adminpc/Desktop/HVK/Script/"
    "Hamiltonian_Vision_Kernel/Main/"
    "data/monalisa.jpg"
)


def build_dataset():

    image = load_image_grayscale(
        img_path )

    patches, positions = extract_patch(
        image,
        patch_size=64
    )

    features = np.array([
        extract_mps_features(p)
        for p in patches
    ])

    features = torch.tensor(
        features,
        dtype=torch.float32
    )

    features = (
        features
        - features.mean(dim=0)
    ) / (
        features.std(dim=0)
        + 1e-8
    )

    positions = sinusoidal_positional_encoding(
        positions,
        d_model=8
    )

    targets = torch.tensor(
        patches,
        dtype=torch.float32
    ).unsqueeze(1)

    return (
        image,
        patches,
        features.to(device),
        positions.to(device),
        targets.to(device)
    )


def train():

    (
        image,
        patches,
        features,
        positions,
        targets
    ) = build_dataset()

    model = QuantumModel(
        feature_dim=features.shape[1],
        positional_dim=positions.shape[1]
    ).to(device)

    decoder = PatchDecoder(
        observable_dim=OBSERVABLE_DIM,
        positional_dim=positions.shape[1],
        patch_size=64
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters())
        +
        list(decoder.parameters()),
        lr=0.003
    )

    total_losses = []
    reconstruction_losses = []
    energy_losses = []

    for step in range(120):

        model.train()

        optimizer.zero_grad()

        observables, energies = model(
            features,
            positions
        )

        output = decoder(
            observables,
            positions
        )

        reconstruction_loss = torch.mean(
            (output - targets) ** 2
        )

        energy_loss = torch.mean(
            energies
        )

        loss = (
            reconstruction_loss
            + 0.01 * energy_loss
        )

        loss.backward()

        optimizer.step()

        total_losses.append(
            loss.item()
        )

        reconstruction_losses.append(
            reconstruction_loss.item()
        )

        energy_losses.append(
            energy_loss.item()
        )

        if step % 20 == 0:

            print(
                f"Step: {step:>4d} | "
                f"Loss: {loss.item():.6f} | "
                f"Recon: {reconstruction_loss.item():.6f} | "
                f"Energy: {energy_loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():

        observables, _ = model(
            features,
            positions
        )

        pred = decoder(
            observables,
            positions
        ).cpu().numpy()

    img_rec = blend_seams(
        stitch(pred)
    )

    mps_patches = np.array([
        [mps_reconstruct(p)]
        for p in patches
    ])

    img_mps = blend_seams(
        stitch(mps_patches)
    )

    plot_reconstructions(
        original=image,
        quantum_reconstruction=img_rec,
        mps_baseline=img_mps,
        random_latent=img_rec,
        zero_latent=img_rec
    )

    entropy_features = (
        features[:, 35:]
        .cpu()
        .numpy()
    )

    plot_entropy_map(
        entropy_features
    )

    plot_observables(
        observables.cpu().numpy()
    )

    plot_training_curves(
        total_losses,
        reconstruction_losses,
        energy_losses
    )

    return model, decoder


if __name__ == "__main__":

    train()