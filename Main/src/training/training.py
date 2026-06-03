import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.preprocessing.image_loader import load_image_grayscale

from src.preprocessing.patching import extract_patches

from src.preprocessing.positional_encoding import (
    sinusoidal_positional_encoding
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


DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "monalisa.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_analysis"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "training_config.json"


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_dataset(
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    image_size: int = 256,
    patch_size: int = 64,
    positional_dim: int = 8,
    device: torch.device | str = "auto",
):
    device = resolve_device(device) if isinstance(device, str) else device
    image = load_image_grayscale(
        str(image_path),
        size=(image_size, image_size)
    )

    patches, positions = extract_patches(
        image,
        patch_size=patch_size
    )
    raw_positions = positions.copy()

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
        features.std(dim=0, unbiased=False)
        + 1e-8
    )

    positions = sinusoidal_positional_encoding(
        positions,
        d_model=positional_dim
    )

    targets = torch.tensor(
        patches,
        dtype=torch.float32
    ).unsqueeze(1)

    return (
        image,
        patches,
        raw_positions,
        features.to(device),
        positions.to(device),
        targets.to(device)
    )


def train(
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    image_size: int = 256,
    patch_size: int = 64,
    positional_dim: int = 8,
    steps: int = 120,
    lr: float = 0.003,
    device: str | torch.device = "auto",
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    save_outputs: bool = True,
    show_plots: bool = False,
):
    device = resolve_device(device) if isinstance(device, str) else device

    (
        image,
        patches,
        raw_positions,
        features,
        positions,
        targets
    ) = build_dataset(
        image_path=image_path,
        image_size=image_size,
        patch_size=patch_size,
        positional_dim=positional_dim,
        device=device,
    )

    model = QuantumModel(
        feature_dim=features.shape[1],
        positional_dim=positions.shape[1]
    ).to(device)

    decoder = PatchDecoder(
        observable_dim=observable_dim,
        positional_dim=positions.shape[1],
        patch_size=patch_size
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters())
        +
        list(decoder.parameters()),
        lr=lr
    )

    total_losses = []
    reconstruction_losses = []
    energy_losses = []

    for step in range(steps):

        model.train()
        decoder.train()

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

        if step % 20 == 0 or step == steps - 1:

            print(
                f"Step: {step:>4d} | "
                f"Loss: {loss.item():.6f} | "
                f"Recon: {reconstruction_loss.item():.6f} | "
                f"Energy: {energy_loss.item():.6f}"
            )

    model.eval()
    decoder.eval()

    with torch.no_grad():

        observables, energies = model(
            features,
            positions
        )

        pred = decoder(
            observables,
            positions
        ).cpu().numpy()

        zero_pred = decoder(
            torch.zeros_like(observables),
            positions
        ).cpu().numpy()

        random_pred = decoder(
            torch.randn_like(observables),
            positions
        ).cpu().numpy()

    img_rec = blend_seams(
        stictch_patches(
            pred,
            image_size=image_size,
            patch_size=patch_size
        ),
        patch_size=patch_size
    )

    mps_patches = np.array([
        [mps_reconstruct(p)]
        for p in patches
    ])

    img_mps = blend_seams(
        stictch_patches(
            mps_patches,
            image_size=image_size,
            patch_size=patch_size
        ),
        patch_size=patch_size
    )

    img_random = blend_seams(
        stictch_patches(
            random_pred,
            image_size=image_size,
            patch_size=patch_size
        ),
        patch_size=patch_size
    )

    img_zero = blend_seams(
        stictch_patches(
            zero_pred,
            image_size=image_size,
            patch_size=patch_size
        ),
        patch_size=patch_size
    )

    history = {
        "total_loss": total_losses,
        "reconstruction_loss": reconstruction_losses,
        "energy_loss": energy_losses,
    }

    outputs = {
        "original": image,
        "quantum_reconstruction": img_rec,
        "mps_baseline": img_mps,
        "random_latent": img_random,
        "zero_latent": img_zero,
        "observables": observables.cpu().numpy(),
        "energies": energies.cpu().numpy(),
        "positions": raw_positions,
        "history": history,
    }

    if save_outputs and output_dir is not None:
        save_analysis_outputs(
            outputs=outputs,
            features=features,
            output_dir=output_dir,
            image_size=image_size,
            patch_size=patch_size,
        )

    if show_plots:
        entropy_features = get_entropy_features(features)

        plot_reconstructions(
            original=image,
            quantum_reconstruction=img_rec,
            mps_baseline=img_mps,
            random_latent=img_random,
            zero_latent=img_zero
        )
        plot_entropy_map(
            entropy_features,
            grid_size=get_grid_size(image_size, patch_size)
        )
        plot_observables(
            observables.cpu().numpy()
        )
        plot_training_curves(
            total_losses,
            reconstruction_losses,
            energy_losses
        )

    return model, decoder, outputs


def get_grid_size(image_size: int, patch_size: int) -> tuple[int, int]:
    return (
        image_size // patch_size,
        image_size // patch_size
    )


def get_entropy_features(features: torch.Tensor) -> np.ndarray:
    feature_array = features.detach().cpu().numpy()
    if feature_array.shape[1] > 35:
        return feature_array[:, 35:]
    return feature_array


def save_analysis_outputs(
    outputs: dict,
    features: torch.Tensor,
    output_dir: str | Path,
    image_size: int,
    patch_size: int,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_reconstruction_plot(
        outputs,
        output_dir / "reconstructions.png"
    )
    save_training_curve(
        outputs["history"],
        output_dir / "training_curves.png"
    )
    save_observable_plot(
        outputs["observables"],
        output_dir / "observables.png"
    )
    save_entropy_plot(
        get_entropy_features(features),
        output_dir / "entropy_map.png",
        grid_size=get_grid_size(image_size, patch_size)
    )

    np.save(
        output_dir / "quantum_reconstruction.npy",
        outputs["quantum_reconstruction"]
    )
    np.save(
        output_dir / "mps_baseline.npy",
        outputs["mps_baseline"]
    )
    np.save(
        output_dir / "observables.npy",
        outputs["observables"]
    )

    metrics = {
        "final_total_loss": outputs["history"]["total_loss"][-1],
        "final_reconstruction_loss": outputs["history"]["reconstruction_loss"][-1],
        "final_energy_loss": outputs["history"]["energy_loss"][-1],
        "mean_energy": float(np.mean(outputs["energies"])),
        "std_energy": float(np.std(outputs["energies"])),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    return metrics


def save_reconstruction_plot(outputs: dict, output_path: Path):
    panels = [
        ("Original", outputs["original"]),
        ("Quantum Reconstruction", outputs["quantum_reconstruction"]),
        ("MPS Baseline", outputs["mps_baseline"]),
        ("Random Latent", outputs["random_latent"]),
        ("Zero Latent", outputs["zero_latent"]),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(22, 5))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("Hamiltonian Vision Kernel", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_curve(history: dict, output_path: Path):
    steps = range(len(history["total_loss"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, history["total_loss"], label="Total Loss")
    ax.plot(steps, history["reconstruction_loss"], label="Reconstruction Loss")
    ax.plot(steps, history["energy_loss"], label="Energy Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("HVK Training Curves")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_observable_plot(observables: np.ndarray, output_path: Path):
    z_obs = observables[:, :6]
    x_obs = observables[:, 6:12]
    zz_corr = observables[:, 12:17]
    xx_corr = observables[:, 17:22]
    yy_corr = observables[:, 22:]

    panels = [
        ("Local Z", z_obs),
        ("Local X", x_obs),
        ("ZZ Correlations", zz_corr),
        ("XX Correlations", xx_corr),
        ("YY Correlations", yy_corr),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    for ax, (title, data) in zip(axes, panels):
        image = ax.imshow(data, aspect="auto", cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046)

    fig.suptitle("Quantum Observable Maps", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_entropy_plot(
    entropy_features: np.ndarray,
    output_path: Path,
    grid_size: tuple[int, int],
):
    patch_entropy = entropy_features.mean(axis=1)
    entropy_map = patch_entropy.reshape(grid_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(entropy_map, cmap="inferno")
    fig.colorbar(image, ax=ax, label="Average Entanglement Entropy")
    ax.set_title("HVK Entanglement Entropy Map")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hamiltonian Vision Kernel training and analysis."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--positional-dim", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "image_path": str(DEFAULT_IMAGE_PATH),
        "image_size": 256,
        "patch_size": 64,
        "positional_dim": 8,
        "steps": 120,
        "lr": 0.003,
        "device": "auto",
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "save_outputs": True,
        "show_plots": False,
    }
    config.update(load_config(args.config))

    for key in [
        "image_path",
        "image_size",
        "patch_size",
        "positional_dim",
        "steps",
        "lr",
        "device",
        "output_dir",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    config["image_path"] = resolve_path(config["image_path"])
    config["output_dir"] = resolve_path(config["output_dir"])

    if args.no_save:
        config["save_outputs"] = False
    if args.show_plots:
        config["show_plots"] = True

    print("Running HVK analysis with config:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    _, _, outputs = train(**config)

    if config["save_outputs"]:
        print(f"Results saved to: {config['output_dir']}")
    print(
        "Final loss:",
        outputs["history"]["total_loss"][-1]
    )


if __name__ == "__main__":

    main()
