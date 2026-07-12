import argparse
import math
import json
import os
import random
import shutil
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
import torch.nn.functional as F
import torch.optim as optim

from src.decoder.patch_decoder import PatchDecoder
from src.preprocessing.image_loader import load_image_grayscale
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.quantum.circuit import observable_dim
from src.quantum.quantum_model import QuantumModel
from src.quantum.symmetric_model import SymmetricQuantumModel
from src.reconstruction.patch_stitching import stictch_patches
from src.reconstruction.seam_bleading import blend_seams
from src.tensornetworks.mps_features import (
    extract_mps_features,
    extract_patch_statistics_features,
)
from src.tensornetworks.mps_reconstruction import mps_reconstruct
from src.training.data_generator import TrainingDataGenerator
from src.training.order_parameters import detect_phase_transition
from src.training.order_parameters import observable_slices
from src.training.phase_media import (
    save_epoch_frame,
    save_frames_as_gif,
    save_merged_phase_transition_gif,
    save_order_parameter_gif,
    save_order_parameter_plot,
    save_phase_transition_order_parameter_gif,
)
from src.visualization.entropy_maps import plot_entropy_map
from src.visualization.observable_plots import plot_observables
from src.visualization.reconstruction_plots import plot_reconstructions
from src.visualization.training_curve import plot_training_curves

DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "monalisa.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_analysis"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "training_config.json"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    feature_mode: str = "mps",
    mps_bond_dim: int = 4,
):
    device = resolve_device(device) if isinstance(device, str) else device
    image = load_image_grayscale(str(image_path), size=(image_size, image_size))

    patches, positions = extract_patches(image, patch_size=patch_size)
    raw_positions = positions.copy()

    if feature_mode == "mps":
        features = np.array(
            [extract_mps_features(p, bond_dim=mps_bond_dim) for p in patches]
        )
    elif feature_mode == "patch-statistics":
        features = np.array([extract_patch_statistics_features(p) for p in patches])
    else:
        raise ValueError("feature_mode must be 'mps' or 'patch-statistics'")

    features = torch.tensor(features, dtype=torch.float32)

    features = (features - features.mean(dim=0)) / (
        features.std(dim=0, unbiased=False) + 1e-8
    )

    positions = sinusoidal_positional_encoding(positions, d_model=positional_dim)

    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    return (
        image,
        patches,
        raw_positions,
        features.to(device),
        positions.to(device),
        targets.to(device),
    )


def train(
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    train_image_paths: list[str | Path] | None = None,
    image_size: int = 256,
    patch_size: int = 64,
    positional_dim: int = 8,
    steps: int = 120,
    lr: float = 0.003,
    device: str | torch.device = "auto",
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    save_outputs: bool = True,
    show_plots: bool = False,
    track_order_parameters: bool = True,
    save_epoch_media: bool = True,
    epoch_frame_interval: int = 1,
    zero_latent_uses_positions: bool = False,
    shuffle_observables_at_eval: bool = False,
    model_variant: str = "standard",
    ablation_mode: str = "baseline",
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
    eval_only_image: str | Path | None = None,
    mps_bond_dim: int = 4,
    qubit_count: int = 6,
    observable_set: str = "full",
    energy_loss_mode: str = "linear",
    energy_weight: float = 0.01,
    energy_margin: float = 0.25,
    log_prefix: str = "",
):
    device = resolve_device(device) if isinstance(device, str) else device
    seed_everything(seed)
    feature_mode = "patch-statistics" if ablation_mode == "no-mps" else "mps"
    dataset_image_path = eval_only_image if eval_only_image is not None else image_path

    (image, patches, raw_positions, features, positions, targets) = build_dataset(
        image_path=dataset_image_path,
        image_size=image_size,
        patch_size=patch_size,
        positional_dim=positional_dim,
        device=device,
        feature_mode=feature_mode,
        mps_bond_dim=mps_bond_dim,
    )
    if train_image_paths:
        train_datasets = [
            build_dataset(
                image_path=path,
                image_size=image_size,
                patch_size=patch_size,
                positional_dim=positional_dim,
                device=device,
                feature_mode=feature_mode,
                mps_bond_dim=mps_bond_dim,
            )
            for path in train_image_paths
        ]
        train_features = torch.cat([dataset[3] for dataset in train_datasets], dim=0)
        train_positions = torch.cat([dataset[4] for dataset in train_datasets], dim=0)
        train_targets = torch.cat([dataset[5] for dataset in train_datasets], dim=0)
    else:
        train_features = features
        train_positions = positions
        train_targets = targets

    model_classes = {
        "standard": QuantumModel,
        "symmetric": SymmetricQuantumModel,
    }
    if model_variant not in model_classes:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. Use one of: {sorted(model_classes)}"
        )
    valid_ablation_modes = {
        "baseline",
        "freeze-classical",
        "freeze-quantum",
        "classical-replacement",
        "classical-matched",
        "random-vqc",
        "no-entanglement",
        "no-energy-loss",
        "no-obs-noise",
        "no-mps",
        "zz-only",
    }
    if ablation_mode not in valid_ablation_modes:
        raise ValueError(
            f"Unknown ablation_mode '{ablation_mode}'. "
            f"Use one of: {sorted(valid_ablation_modes)}"
        )
    valid_energy_loss_modes = {"linear", "positive", "contrastive"}
    if energy_loss_mode not in valid_energy_loss_modes:
        raise ValueError(
            f"Unknown energy_loss_mode '{energy_loss_mode}'. "
            f"Use one of: {sorted(valid_energy_loss_modes)}"
        )
    if (
        ablation_mode in {"classical-replacement", "classical-matched"}
        and model_variant != "standard"
    ):
        raise ValueError(
            f"{ablation_mode} is only available for the standard model."
        )

    model_kwargs = {
        "feature_dim": train_features.shape[1],
        "positional_dim": train_positions.shape[1],
    }
    if model_variant == "standard":
        model_kwargs["qubit_count"] = qubit_count
        model_kwargs["observable_set"] = (
            "zz-only" if ablation_mode == "zz-only" else observable_set
        )
        model_kwargs["use_classical_replacement"] = (
            ablation_mode == "classical-replacement"
        )
        model_kwargs["use_parameter_matched_classical"] = (
            ablation_mode == "classical-matched"
        )
        model_kwargs["vqc_mode"] = {
            "random-vqc": "random",
            "no-entanglement": "no-entanglement",
        }.get(ablation_mode, "standard")
        model_kwargs["observable_noise"] = ablation_mode != "no-obs-noise"
    model = model_classes[model_variant](**model_kwargs).to(device)

    decoder = PatchDecoder(
        observable_dim=model.observable_dim,
        positional_dim=positions.shape[1],
        patch_size=patch_size,
    ).to(device)

    if ablation_mode == "freeze-classical":
        for param in decoder.parameters():
            param.requires_grad_(False)
        model.feature_projection.requires_grad_(False)
        model.position_projection.requires_grad_(False)
    elif ablation_mode == "freeze-quantum":
        for parameter_name in ("weights", "Jx", "Jy", "Jz"):
            parameter = getattr(model, parameter_name, None)
            if parameter is not None:
                parameter.requires_grad_(False)

    trainable_parameters = [
        p
        for p in list(model.parameters()) + list(decoder.parameters())
        if p.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(f"Ablation mode '{ablation_mode}' left no trainable parameters.")
    optimizer = optim.Adam(trainable_parameters, lr=lr)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if eval_only_image is not None:
        if checkpoint_dir is None:
            raise ValueError("--eval-only-image requires --checkpoint-dir")
        model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", map_location=device)
        )
        decoder.load_state_dict(
            torch.load(checkpoint_dir / "decoder.pt", map_location=device)
        )
        steps = 0

    total_losses = []
    reconstruction_losses = []
    energy_losses = []
    epoch_frame_paths = []
    data_generator = None
    frame_dir = None
    if save_outputs and output_dir is not None and track_order_parameters:
        output_dir = Path(output_dir)
        data_generator = TrainingDataGenerator(output_dir, Path(image_path).name)
        if save_epoch_media:
            frame_dir = output_dir / "phase_transition_frames"
            if frame_dir.exists():
                shutil.rmtree(frame_dir)

    for step in range(steps):
        model.train()
        decoder.train()

        optimizer.zero_grad()

        observables, energies = model(train_features, train_positions)

        output = decoder(observables, train_positions)

        reconstruction_loss = torch.mean((output - train_targets) ** 2)

        if energy_loss_mode == "linear":
            energy_loss = torch.mean(energies)
        elif energy_loss_mode == "positive":
            energy_loss = torch.mean(energies.square())
        else:
            permutation = torch.randperm(
                train_positions.shape[0], device=train_positions.device
            )
            _, negative_energies = model(train_features, train_positions[permutation])
            energy_loss = F.softplus(
                energies - negative_energies + energy_margin
            ).mean()

        if ablation_mode == "no-energy-loss":
            loss = reconstruction_loss
        else:
            loss = reconstruction_loss + energy_weight * energy_loss

        loss.backward()

        optimizer.step()

        total_losses.append(loss.item())

        reconstruction_losses.append(reconstruction_loss.item())

        energy_losses.append(energy_loss.item())

        if step % 20 == 0 or step == steps - 1:
            prefix = f"[{log_prefix}] " if log_prefix else ""
            print(
                f"{prefix}Step: {step:>4d} | "
                f"Loss: {loss.item():.6f} | "
                f"Recon: {reconstruction_loss.item():.6f} | "
                f"Energy: {energy_loss.item():.6f}"
            )

        should_track = data_generator is not None and (
            step % max(epoch_frame_interval, 1) == 0 or step == steps - 1
        )
        if should_track:
            model.eval()
            decoder.eval()
            with torch.no_grad():
                tracked_observables, tracked_energies = model(features, positions)
                tracked_pred = decoder(tracked_observables, positions).cpu().numpy()
            tracked_reconstruction = blend_seams(
                stictch_patches(
                    tracked_pred, image_size=image_size, patch_size=patch_size
                ),
                patch_size=patch_size,
            )
            frame_path = ""
            if frame_dir is not None:
                quick_summary = data_generator.record(
                    epoch=step,
                    step=step,
                    frame_path="",
                    total_loss=loss.item(),
                    reconstruction_mse=reconstruction_loss.item(),
                    observables=tracked_observables.cpu().numpy(),
                    energies=tracked_energies.cpu().numpy(),
                    positions=raw_positions,
                )
                saved_frame = save_epoch_frame(
                    original=image,
                    reconstruction=tracked_reconstruction,
                    epoch=step,
                    total_loss=loss.item(),
                    order_parameter=quick_summary["mean_order_parameter"],
                    output_dir=frame_dir,
                )
                frame_path = str(saved_frame)
                epoch_frame_paths.append(saved_frame)
                data_generator.epoch_rows[-1]["reconstructed_image"] = frame_path
            else:
                data_generator.record(
                    epoch=step,
                    step=step,
                    frame_path=frame_path,
                    total_loss=loss.item(),
                    reconstruction_mse=reconstruction_loss.item(),
                    observables=tracked_observables.cpu().numpy(),
                    energies=tracked_energies.cpu().numpy(),
                    positions=raw_positions,
                )
            model.train()
            decoder.train()

    model.eval()
    decoder.eval()

    with torch.no_grad():
        observables, energies = model(features, positions)

        pred = decoder(observables, positions).cpu().numpy()

        shuffled_pred = None
        shuffle_metadata = None
        if shuffle_observables_at_eval:
            perm = torch.randperm(observables.shape[0], device=observables.device)
            identity = torch.arange(observables.shape[0], device=observables.device)
            fixed_points = int(torch.sum(perm == identity).item())
            shuffle_metadata = {
                "permutation": perm.detach().cpu().tolist(),
                "num_observables": int(observables.shape[0]),
                "fixed_points": fixed_points,
                "changed_points": int(observables.shape[0] - fixed_points),
                "is_identity": bool(fixed_points == observables.shape[0]),
            }
            shuffled_pred = decoder(observables[perm], positions).cpu().numpy()

        zero_positions = (
            positions if zero_latent_uses_positions else torch.zeros_like(positions)
        )

        zero_pred = decoder(torch.zeros_like(observables), zero_positions).cpu().numpy()

        random_pred = decoder(torch.randn_like(observables), positions).cpu().numpy()

    img_rec = blend_seams(
        stictch_patches(pred, image_size=image_size, patch_size=patch_size),
        patch_size=patch_size,
    )

    mps_patches = np.array([[mps_reconstruct(p)] for p in patches])

    img_mps = blend_seams(
        stictch_patches(mps_patches, image_size=image_size, patch_size=patch_size),
        patch_size=patch_size,
    )

    img_random = blend_seams(
        stictch_patches(random_pred, image_size=image_size, patch_size=patch_size),
        patch_size=patch_size,
    )

    img_zero = blend_seams(
        stictch_patches(zero_pred, image_size=image_size, patch_size=patch_size),
        patch_size=patch_size,
    )

    img_shuffled = None
    if shuffled_pred is not None:
        img_shuffled = blend_seams(
            stictch_patches(shuffled_pred, image_size=image_size, patch_size=patch_size),
            patch_size=patch_size,
        )

    history = {
        "total_loss": total_losses,
        "reconstruction_loss": reconstruction_losses,
        "energy_loss": energy_losses,
    }
    comparison_metrics = {
        "quantum_reconstruction": compute_image_metrics(img_rec, image),
        "mps_baseline": compute_image_metrics(img_mps, image),
        "random_latent": compute_image_metrics(img_random, image),
        "zero_latent": compute_image_metrics(img_zero, image),
    }
    if img_shuffled is not None:
        comparison_metrics["shuffled_observables"] = compute_image_metrics(
            img_shuffled, image
        )

    outputs = {
        "original": image,
        "quantum_reconstruction": img_rec,
        "mps_baseline": img_mps,
        "random_latent": img_random,
        "zero_latent": img_zero,
        "shuffled_observables": img_shuffled,
        "observables": observables.cpu().numpy(),
        "energies": energies.cpu().numpy(),
        "positions": raw_positions,
        "history": history,
        "comparison_metrics": comparison_metrics,
        "reconstruction_metrics": comparison_metrics["quantum_reconstruction"],
        "media": {},
        "epoch_order_parameters": data_generator.epoch_rows if data_generator else [],
        "phase_transition": (
            detect_phase_transition(data_generator.epoch_rows)
            if data_generator
            else detect_phase_transition([])
        ),
        "model_variant": model_variant,
        "ablation_mode": ablation_mode,
        "seed": seed,
        "feature_mode": feature_mode,
        "mps_bond_dim": mps_bond_dim,
        "qubit_count": qubit_count,
        "observable_set": getattr(model, "observable_set", "full"),
        "energy_loss_mode": energy_loss_mode,
        "energy_weight": energy_weight,
        "energy_margin": energy_margin,
        "eval_only_image": str(eval_only_image) if eval_only_image is not None else None,
        "train_image_paths": (
            [str(path) for path in train_image_paths] if train_image_paths else None
        ),
        "zero_latent_uses_positions": zero_latent_uses_positions,
        "shuffle_metadata": shuffle_metadata,
    }

    if save_outputs and output_dir is not None:
        save_analysis_outputs(
            outputs=outputs,
            features=features,
            output_dir=output_dir,
            image_size=image_size,
            patch_size=patch_size,
            data_generator=data_generator,
            epoch_frame_paths=epoch_frame_paths,
            save_epoch_media=save_epoch_media,
            model=model,
            decoder=decoder,
            seed=seed,
            eval_target=image,
        )

    if show_plots:
        entropy_features = get_entropy_features(features)

        plot_reconstructions(
            original=image,
            quantum_reconstruction=img_rec,
            mps_baseline=img_mps,
            random_latent=img_random,
            zero_latent=img_zero,
        )
        plot_entropy_map(
            entropy_features, grid_size=get_grid_size(image_size, patch_size)
        )
        plot_observables(observables.cpu().numpy())
        plot_training_curves(total_losses, reconstruction_losses, energy_losses)

    return model, decoder, outputs


def get_grid_size(image_size: int, patch_size: int) -> tuple[int, int]:
    return (image_size // patch_size, image_size // patch_size)


def get_entropy_features(features: torch.Tensor) -> np.ndarray:
    feature_array = features.detach().cpu().numpy()
    if feature_array.shape[1] > 35:
        return feature_array[:, 35:]
    return feature_array


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def psnr_from_mse(value: float) -> float:
    if value <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(value))


def simple_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    cov = float(((x - mu_x) * (y - mu_y)).mean())
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator)


def compute_image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    value = mse(prediction, target)
    return {
        "mse": value,
        "psnr": psnr_from_mse(value),
        "ssim": simple_ssim(prediction, target),
    }


def save_analysis_outputs(
    outputs: dict,
    features: torch.Tensor,
    output_dir: str | Path,
    image_size: int,
    patch_size: int,
    data_generator: TrainingDataGenerator | None = None,
    epoch_frame_paths: list[Path] | None = None,
    save_epoch_media: bool = True,
    model: torch.nn.Module | None = None,
    decoder: torch.nn.Module | None = None,
    seed: int | None = None,
    eval_target: np.ndarray | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_reconstruction_plot(outputs, output_dir / "reconstructions.png")
    save_training_curve(outputs["history"], output_dir / "training_curves.png")
    save_observable_plot(outputs["observables"], output_dir / "observables.png")
    save_entropy_plot(
        get_entropy_features(features),
        output_dir / "entropy_map.png",
        grid_size=get_grid_size(image_size, patch_size),
    )
    if data_generator is not None:
        data_generator.write_csvs()
        media_paths = outputs.setdefault("media", {})
        order_curve_path = save_order_parameter_plot(
            data_generator.epoch_rows,
            output_dir / "hvk_order_parameter_curve.png",
        )
        if order_curve_path is not None:
            media_paths["order_parameter_curve"] = str(order_curve_path)
        if save_epoch_media:
            phase_order_gif_path = save_phase_transition_order_parameter_gif(
                data_generator.epoch_rows,
                outputs["phase_transition"],
                output_dir / "phase_transition_epoch_vs_order_parameter.gif",
            )
            if phase_order_gif_path is not None:
                media_paths["phase_transition_epoch_vs_order_parameter_gif"] = str(
                    phase_order_gif_path
                )
            merged_gif_path = save_merged_phase_transition_gif(
                data_generator.epoch_rows,
                outputs["phase_transition"],
                epoch_frame_paths or [],
                output_dir / "phase_transition_order_parameter_reconstruction.gif",
            )
            if merged_gif_path is not None:
                media_paths[
                    "phase_transition_order_parameter_reconstruction_gif"
                ] = str(merged_gif_path)
            order_signal_gif_path = save_order_parameter_gif(
                data_generator.epoch_rows,
                outputs["phase_transition"],
                output_dir / "hvk_order_parameter_phase_transition.gif",
            )
            if order_signal_gif_path is not None:
                media_paths["order_parameter_phase_transition_gif"] = str(
                    order_signal_gif_path
                )
            reconstruction_gif_path = save_frames_as_gif(
                epoch_frame_paths or [],
                output_dir / "hvk_reconstruction_phase_transition.gif",
            )
            if reconstruction_gif_path is not None:
                media_paths["reconstruction_phase_transition_gif"] = str(
                    reconstruction_gif_path
                )

    np.save(
        output_dir / "quantum_reconstruction.npy", outputs["quantum_reconstruction"]
    )
    np.save(output_dir / "mps_baseline.npy", outputs["mps_baseline"])
    np.save(output_dir / "random_latent.npy", outputs["random_latent"])
    np.save(output_dir / "zero_latent.npy", outputs["zero_latent"])
    np.save(output_dir / "observables.npy", outputs["observables"])
    if outputs.get("shuffled_observables") is not None:
        np.save(output_dir / "shuffled_observables.npy", outputs["shuffled_observables"])

    target = outputs["original"] if eval_target is None else eval_target
    reconstruction_metrics = compute_image_metrics(
        outputs["quantum_reconstruction"], target
    )
    comparison_metrics = {
        "quantum_reconstruction": reconstruction_metrics,
        "mps_baseline": compute_image_metrics(outputs["mps_baseline"], target),
        "random_latent": compute_image_metrics(outputs["random_latent"], target),
        "zero_latent": compute_image_metrics(outputs["zero_latent"], target),
    }
    if outputs.get("shuffled_observables") is not None:
        comparison_metrics["shuffled_observables"] = compute_image_metrics(
            outputs["shuffled_observables"], target
        )
    total_history = outputs["history"]["total_loss"]
    reconstruction_history = outputs["history"]["reconstruction_loss"]
    energy_history = outputs["history"]["energy_loss"]
    metrics = {
        "final_total_loss": total_history[-1] if total_history else None,
        "final_reconstruction_loss": (
            reconstruction_history[-1] if reconstruction_history else None
        ),
        "final_energy_loss": energy_history[-1] if energy_history else None,
        "reconstruction_metrics": reconstruction_metrics,
        "comparison_metrics": comparison_metrics,
        "mse": reconstruction_metrics["mse"],
        "psnr": reconstruction_metrics["psnr"],
        "ssim": reconstruction_metrics["ssim"],
        "mean_energy": float(np.mean(outputs["energies"])),
        "std_energy": float(np.std(outputs["energies"])),
        "phase_transition": outputs["phase_transition"],
        "media": outputs.get("media", {}),
        "model_variant": outputs.get("model_variant", "standard"),
        "ablation_mode": outputs.get("ablation_mode", "baseline"),
        "feature_mode": outputs.get("feature_mode", "mps"),
        "mps_bond_dim": outputs.get("mps_bond_dim"),
        "qubit_count": outputs.get("qubit_count"),
        "observable_set": outputs.get("observable_set"),
        "energy_loss_mode": outputs.get("energy_loss_mode", "linear"),
        "energy_weight": outputs.get("energy_weight"),
        "energy_margin": outputs.get("energy_margin"),
        "seed": outputs.get("seed", seed),
        "eval_only_image": outputs.get("eval_only_image"),
        "train_image_paths": outputs.get("train_image_paths"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if outputs.get("shuffled_observables") is not None:
        shuffled_summary = {
            "normal_mse_vs_original": comparison_metrics[
                "quantum_reconstruction"
            ]["mse"],
            "normal_psnr_vs_original": comparison_metrics[
                "quantum_reconstruction"
            ]["psnr"],
            "normal_ssim_vs_original": comparison_metrics[
                "quantum_reconstruction"
            ]["ssim"],
            "shuffled_mse_vs_original": comparison_metrics[
                "shuffled_observables"
            ]["mse"],
            "shuffled_psnr_vs_original": comparison_metrics[
                "shuffled_observables"
            ]["psnr"],
            "shuffled_ssim_vs_original": comparison_metrics[
                "shuffled_observables"
            ]["ssim"],
            "shuffled_mse_vs_normal": mse(
                outputs["shuffled_observables"],
                outputs["quantum_reconstruction"],
            ),
            "normal_shape": list(outputs["quantum_reconstruction"].shape),
            "shuffled_shape": list(outputs["shuffled_observables"].shape),
            "shuffle_metadata": outputs.get("shuffle_metadata"),
        }
        (output_dir / "shuffle_eval_summary.json").write_text(
            json.dumps(shuffled_summary, indent=2), encoding="utf-8"
        )
    if outputs.get("zero_latent_uses_positions"):
        zero_summary = {
            "normal_mse_vs_original": comparison_metrics[
                "quantum_reconstruction"
            ]["mse"],
            "normal_psnr_vs_original": comparison_metrics[
                "quantum_reconstruction"
            ]["psnr"],
            "normal_ssim_vs_original": comparison_metrics[
                "quantum_reconstruction"
            ]["ssim"],
            "zero_latent_mse_vs_original": comparison_metrics["zero_latent"][
                "mse"
            ],
            "zero_latent_psnr_vs_original": comparison_metrics["zero_latent"][
                "psnr"
            ],
            "zero_latent_ssim_vs_original": comparison_metrics["zero_latent"][
                "ssim"
            ],
            "zero_latent_mse_vs_normal": mse(
                outputs["zero_latent"],
                outputs["quantum_reconstruction"],
            ),
            "random_latent_mse_vs_original": comparison_metrics[
                "random_latent"
            ]["mse"],
            "random_latent_psnr_vs_original": comparison_metrics[
                "random_latent"
            ]["psnr"],
            "random_latent_ssim_vs_original": comparison_metrics[
                "random_latent"
            ]["ssim"],
            "normal_shape": list(outputs["quantum_reconstruction"].shape),
            "zero_latent_shape": list(outputs["zero_latent"].shape),
        }
        (output_dir / "zero_latent_eval_summary.json").write_text(
            json.dumps(zero_summary, indent=2), encoding="utf-8"
        )
    if model is not None and decoder is not None:
        torch.save(model.state_dict(), output_dir / "model.pt")
        torch.save(decoder.state_dict(), output_dir / "decoder.pt")

    return metrics


def save_reconstruction_plot(outputs: dict, output_path: Path):
    panels = [
        ("Original", outputs["original"]),
        ("Quantum Reconstruction", outputs["quantum_reconstruction"]),
        ("MPS Baseline", outputs["mps_baseline"]),
        ("Random Latent", outputs["random_latent"]),
        ("Zero Latent", outputs["zero_latent"]),
    ]
    if outputs.get("shuffled_observables") is not None:
        panels.append(("Shuffled Observables", outputs["shuffled_observables"]))

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
    parts = observable_slices(observables)

    panels = [
        ("Local Z", parts["z"]),
        ("Local X", parts["x"]),
        ("ZZ Correlations", parts["zz"]),
    ]
    if np.any(parts["xx"]) or np.any(parts["yy"]):
        panels.extend(
            [
                ("XX Correlations", parts["xx"]),
                ("YY Correlations", parts["yy"]),
            ]
        )

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
    parser.add_argument("--train-image-paths", type=Path, nargs="+")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--positional-dim", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--no-order-tracking", action="store_true")
    parser.add_argument("--no-epoch-media", action="store_true")
    parser.add_argument("--epoch-frame-interval", type=int)
    parser.add_argument("--zero-latent-uses-positions", action="store_true")
    parser.add_argument("--shuffle-observables-at-eval", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--eval-only-image", type=Path)
    parser.add_argument("--mps-bond-dim", type=int)
    parser.add_argument("--qubit-count", type=int)
    parser.add_argument("--observable-set", choices=["full", "zz-only"])
    parser.add_argument(
        "--energy-loss-mode",
        choices=["linear", "positive", "contrastive"],
        default=None,
    )
    parser.add_argument("--energy-weight", type=float)
    parser.add_argument("--energy-margin", type=float)
    parser.add_argument(
        "--model-variant",
        choices=["standard", "symmetric"],
        default=None,
    )
    parser.add_argument(
        "--ablation-mode",
        choices=[
            "baseline",
            "freeze-classical",
            "freeze-quantum",
            "classical-replacement",
            "classical-matched",
            "random-vqc",
            "no-entanglement",
            "no-energy-loss",
            "no-obs-noise",
            "no-mps",
            "zz-only",
        ],
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "image_path": str(DEFAULT_IMAGE_PATH),
        "train_image_paths": None,
        "image_size": 256,
        "patch_size": 64,
        "positional_dim": 8,
        "steps": 120,
        "lr": 0.003,
        "device": "auto",
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "save_outputs": True,
        "show_plots": False,
        "track_order_parameters": True,
        "save_epoch_media": True,
        "epoch_frame_interval": 1,
        "zero_latent_uses_positions": False,
        "shuffle_observables_at_eval": False,
        "model_variant": "standard",
        "ablation_mode": "baseline",
        "seed": 42,
        "checkpoint_dir": None,
        "eval_only_image": None,
        "mps_bond_dim": 4,
        "qubit_count": 6,
        "observable_set": "full",
        "energy_loss_mode": "linear",
        "energy_weight": 0.01,
        "energy_margin": 0.25,
    }
    config.update(load_config(args.config))

    for key in [
        "image_path",
        "train_image_paths",
        "image_size",
        "patch_size",
        "positional_dim",
        "steps",
        "lr",
        "device",
        "output_dir",
        "seed",
        "checkpoint_dir",
        "eval_only_image",
        "mps_bond_dim",
        "qubit_count",
        "observable_set",
        "energy_loss_mode",
        "energy_weight",
        "energy_margin",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    config["image_path"] = resolve_path(config["image_path"])
    if config.get("train_image_paths") is not None:
        config["train_image_paths"] = [
            resolve_path(path) for path in config["train_image_paths"]
        ]
    config["output_dir"] = resolve_path(config["output_dir"])

    if args.no_save:
        config["save_outputs"] = False
    if args.show_plots:
        config["show_plots"] = True
    if args.no_order_tracking:
        config["track_order_parameters"] = False
    if args.no_epoch_media:
        config["save_epoch_media"] = False
    if args.epoch_frame_interval is not None:
        config["epoch_frame_interval"] = args.epoch_frame_interval
    if args.zero_latent_uses_positions:
        config["zero_latent_uses_positions"] = True
    if args.shuffle_observables_at_eval:
        config["shuffle_observables_at_eval"] = True
    if args.model_variant is not None:
        config["model_variant"] = args.model_variant
    if args.ablation_mode is not None:
        config["ablation_mode"] = args.ablation_mode

    print("Running HVK analysis with config:")
    print(json.dumps({k: str(v) for k, v in config.items()}, indent=2))

    _, _, outputs = train(**config)

    if config["save_outputs"]:
        print(f"Results saved to: {config['output_dir']}")
    final_loss = (
        outputs["history"]["total_loss"][-1]
        if outputs["history"]["total_loss"]
        else None
    )
    print("Final loss:", final_loss)


if __name__ == "__main__":
    main()
