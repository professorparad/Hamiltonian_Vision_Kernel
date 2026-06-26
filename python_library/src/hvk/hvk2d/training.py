from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from hvk.config import HVK2DConfig
from hvk.hvk2d.model import PatchDecoder2D, Quantum2DGridModel
from hvk.hvk2d.order_parameters import hvk2d_correlation_rows, hvk2d_order_summary
from hvk.preprocessing.image_loader import load_image_grayscale
from hvk.preprocessing.patching import extract_patches
from hvk.preprocessing.positional_encoding import sinusoidal_positional_encoding
from hvk.reconstruction.patch_stitching import stictch_patches
from hvk.reconstruction.seam_bleading import blend_seams
from hvk.tensornetworks.mps_features import extract_mps_features
from hvk.training.order_parameters import detect_phase_transition
from hvk.training.phase_media import save_order_parameter_plot
from hvk.training.training import resolve_device, save_reconstruction_plot, save_training_curve


def run_hvk2d(config: HVK2DConfig | None = None, **overrides):
    if config is None:
        config = HVK2DConfig(**overrides)
    elif overrides:
        config = HVK2DConfig(**{**config.__dict__, **overrides})
    config.validate()

    device = resolve_device(config.device, requires_quantum=True)
    data = build_hvk2d_dataset(config, device)
    model = Quantum2DGridModel(
        feature_dim=data["features"].shape[1],
        positional_dim=data["positions"].shape[1],
    ).to(device)
    decoder = PatchDecoder2D(
        positional_dim=data["positions"].shape[1],
        patch_size=config.patch_size,
    ).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=config.lr)
    output_dir = Path(config.output_dir) if config.output_dir is not None else None
    epoch_rows = []
    correlation_rows = []
    previous_order = None
    history = {"total_loss": [], "reconstruction_loss": [], "energy_loss": []}

    for step in range(config.steps):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(data["features"], data["positions"])
        output = decoder(observables, data["positions"])
        reconstruction_loss = torch.mean((output - data["targets"]) ** 2)
        energy_loss = torch.mean(energies)
        loss = reconstruction_loss + 0.01 * energy_loss
        loss.backward()
        optimizer.step()
        history["total_loss"].append(loss.item())
        history["reconstruction_loss"].append(reconstruction_loss.item())
        history["energy_loss"].append(energy_loss.item())

        should_record = step % max(config.epoch_frame_interval, 1) == 0 or step == config.steps - 1
        if config.track_order_parameters and should_record:
            obs_np = observables.detach().cpu().numpy()
            energy_np = energies.detach().cpu().numpy()
            summary_fn = config.order_parameter_fn or hvk2d_order_summary
            summary = summary_fn(obs_np, energy_np, previous_order)
            previous_order = summary["mean_order_parameter"]
            epoch_rows.append(
                {
                    "image": Path(config.image_path).name,
                    "epoch": step,
                    "step": step,
                    "total_loss": loss.item(),
                    "reconstruction_mse": reconstruction_loss.item(),
                    **summary,
                }
            )
            correlation_rows.extend(
                hvk2d_correlation_rows(
                    image_name=Path(config.image_path).name,
                    epoch=step,
                    step=step,
                    total_loss=loss.item(),
                    reconstruction_mse=reconstruction_loss.item(),
                    observables=obs_np,
                    energies=energy_np,
                    positions=data["raw_positions"],
                )
            )

    model.eval()
    decoder.eval()
    with torch.no_grad():
        observables, energies = model(data["features"], data["positions"])
        pred = decoder(observables, data["positions"]).cpu().numpy()
    reconstruction = blend_seams(
        stictch_patches(pred, image_size=config.image_size, patch_size=config.patch_size),
        patch_size=config.patch_size,
    )
    phase_transition = detect_phase_transition(epoch_rows)
    outputs = {
        "model": model,
        "decoder": decoder,
        "original": data["image"],
        "quantum_reconstruction": reconstruction,
        "observables": observables.cpu().numpy(),
        "energies": energies.cpu().numpy(),
        "history": history,
        "epoch_order_parameters": epoch_rows,
        "correlation_rows": correlation_rows,
        "phase_transition": phase_transition,
        "config": config,
        "media": {},
    }
    if config.save_outputs and output_dir is not None:
        save_hvk2d_outputs(outputs, output_dir)
    return outputs


def build_hvk2d_dataset(config: HVK2DConfig, device: torch.device):
    image = load_image_grayscale(str(config.image_path), size=(config.image_size, config.image_size))
    patches, raw_positions = extract_patches(image, patch_size=config.patch_size)
    features = np.array([extract_mps_features(patch) for patch in patches])
    features = torch.tensor(features, dtype=torch.float32)
    features = (features - features.mean(dim=0)) / (features.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=config.positional_dim)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    return {
        "image": image,
        "patches": patches,
        "raw_positions": raw_positions,
        "features": features.to(device),
        "positions": positions.to(device),
        "targets": targets.to(device),
    }


def save_hvk2d_outputs(outputs: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_reconstruction_plot(outputs, output_dir / "reconstructions.png")
    save_training_curve(outputs["history"], output_dir / "training_curves.png")
    order_curve = save_order_parameter_plot(
        outputs["epoch_order_parameters"],
        output_dir / "hvk2d_order_parameter_curve.png",
    )
    if order_curve is not None:
        outputs["media"]["order_parameter_curve"] = str(order_curve)
    write_csv(output_dir / "hvk2d_epoch_order_parameters.csv", outputs["epoch_order_parameters"])
    write_csv(output_dir / "hvk2d_epoch_correlation_table.csv", outputs["correlation_rows"])


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
