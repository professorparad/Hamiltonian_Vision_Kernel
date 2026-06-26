from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.optim as optim

from Main2.src.analysis import (
    detect_phase_transition,
    main2_correlation_rows,
    main2_order_summary,
)
from Main2.src.config import Main2Config
from Main2.src.dataset import build_main2_dataset
from Main2.src.model import PatchDecoder, Quantum2DGridModel
from Main2.src.outputs import (
    save_gif,
    save_order_curve,
    save_reconstruction_frame,
    write_csv,
)
from Main2.src.pathing import add_main_package_to_path

add_main_package_to_path()

from src.reconstruction.patch_stitching import stictch_patches
from src.reconstruction.seam_bleading import blend_seams
from src.training.phase_media import (
    save_merged_phase_transition_gif,
    save_phase_transition_order_parameter_gif,
)
from src.training.training import resolve_device


def run_main2(config: Main2Config):
    device = resolve_device(config.device, requires_quantum=True)
    data = build_main2_dataset(
        image_path=config.image_path,
        image_size=config.image_size,
        patch_size=config.patch_size,
        positional_dim=config.positional_dim,
        device=device,
    )
    model = Quantum2DGridModel(
        feature_dim=data["features"].shape[1],
        positional_dim=data["positions"].shape[1],
    ).to(device)
    decoder = PatchDecoder(
        positional_dim=data["positions"].shape[1],
        patch_size=config.patch_size,
    ).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=config.lr)
    output_dir = Path(config.output_dir)
    frame_dir = output_dir / "phase_transition_frames"
    epoch_rows = []
    correlation_rows = []
    frame_paths = []
    previous_order = None

    for step in range(config.steps + 1):
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

        if step % 20 == 0:
            print(
                f"Step: {step:>4d} | Total Loss: {loss.item():.6f} | "
                f"Reconstruction MSE: {reconstruction_loss.item():.6f} | "
                f"2D Energy: {energy_loss.item():.6f}"
            )

        should_record = step % max(config.frame_interval, 1) == 0 or step == config.steps
        if should_record:
            model.eval()
            decoder.eval()
            with torch.no_grad():
                tracked_observables, tracked_energies = model(
                    data["features"], data["positions"]
                )
                pred_patches = decoder(
                    tracked_observables, data["positions"]
                ).cpu().numpy()
            reconstruction = blend_seams(
                stictch_patches(
                    pred_patches,
                    image_size=config.image_size,
                    patch_size=config.patch_size,
                ),
                patch_size=config.patch_size,
            )
            frame_path = save_reconstruction_frame(
                data["image"], reconstruction, step, frame_dir
            )
            frame_paths.append(frame_path)
            obs_np = tracked_observables.cpu().numpy()
            energy_np = tracked_energies.cpu().numpy()
            summary = main2_order_summary(obs_np, energy_np, previous_order)
            previous_order = summary["mean_order_parameter"]
            epoch_rows.append(
                {
                    "image": Path(config.image_path).name,
                    "epoch": step,
                    "step": step,
                    "reconstructed_image": str(frame_path),
                    "total_loss": loss.item(),
                    "reconstruction_mse": reconstruction_loss.item(),
                    **summary,
                }
            )
            correlation_rows.extend(
                main2_correlation_rows(
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

    phase_transition = detect_phase_transition(epoch_rows)

    write_csv(output_dir / "hvk_epoch_reconstruction_table.csv", epoch_rows)
    write_csv(output_dir / "hvk_epoch_correlation_table.csv", correlation_rows)
    save_order_curve(epoch_rows, output_dir / "hvk_order_parameter_curve.png")
    gif_path = None
    order_gif_path = None
    merged_gif_path = None
    if config.save_gif:
        gif_path = save_gif(
            frame_paths, output_dir / "hvk_reconstruction_phase_transition.gif"
        )
        order_gif_path = save_phase_transition_order_parameter_gif(
            epoch_rows,
            phase_transition,
            output_dir / "phase_transition_epoch_vs_order_parameter.gif",
        )
        merged_gif_path = save_merged_phase_transition_gif(
            epoch_rows,
            phase_transition,
            frame_paths,
            output_dir / "phase_transition_order_parameter_reconstruction.gif",
        )
    return {
        "model": model,
        "decoder": decoder,
        "epoch_rows": epoch_rows,
        "correlation_rows": correlation_rows,
        "frame_paths": frame_paths,
        "gif_path": gif_path,
        "order_gif_path": order_gif_path,
        "merged_gif_path": merged_gif_path,
        "phase_transition": phase_transition,
    }
