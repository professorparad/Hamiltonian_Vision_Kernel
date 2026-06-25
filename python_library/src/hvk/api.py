"""Public API for the Hamiltonian Vision Kernel package."""
from __future__ import annotations

import argparse
from pathlib import Path

from hvk.config import HVK2DConfig, HVKRunConfig
from hvk.hvk2d.training import run_hvk2d as _run_hvk2d
from hvk.training.training import train

MODEL_VARIANTS = ("standard", "symmetric")


def run_hvk1d(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    image_size: int = 256,
    patch_size: int = 64,
    positional_dim: int = 8,
    steps: int = 120,
    lr: float = 0.003,
    device: str = "auto",
    model_variant: str = "standard",
    save_outputs: bool = True,
    show_plots: bool = False,
    track_order_parameters: bool = True,
    save_epoch_media: bool = True,
    epoch_frame_interval: int = 1,
    zero_latent_uses_positions: bool = False,
) -> dict:
    """Train one HVK1D variant and return the training outputs."""
    config = HVKRunConfig(
        image_path=image_path,
        output_dir=output_dir,
        image_size=image_size,
        patch_size=patch_size,
        positional_dim=positional_dim,
        steps=steps,
        lr=lr,
        device=device,
        image_mode="grayscale",
        encoding="sinusoidal",
        save_outputs=save_outputs,
        show_plots=show_plots,
        track_order_parameters=track_order_parameters,
        save_epoch_media=save_epoch_media,
        epoch_frame_interval=epoch_frame_interval,
        zero_latent_uses_positions=zero_latent_uses_positions,
    )
    config.validate()
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}, got {model_variant!r}")

    _, _, outputs = train(
        image_path=image_path,
        image_size=image_size,
        patch_size=patch_size,
        positional_dim=positional_dim,
        steps=steps,
        lr=lr,
        device=device,
        output_dir=output_dir,
        save_outputs=save_outputs,
        show_plots=show_plots,
        track_order_parameters=track_order_parameters,
        save_epoch_media=save_epoch_media,
        epoch_frame_interval=epoch_frame_interval,
        zero_latent_uses_positions=zero_latent_uses_positions,
        model_variant=model_variant,
    )
    return outputs


def run_hvk2d(config: HVK2DConfig | None = None, **overrides) -> dict:
    """Train the 2D grid HVK model and return outputs."""
    return _run_hvk2d(config, **overrides)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HVK1D image reconstruction.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("hvk_outputs"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--positional-dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="standard")
    parser.add_argument("--variant-family", choices=["hvk1d", "hvk2d"], default="hvk1d")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-order-tracking", action="store_true")
    parser.add_argument("--no-epoch-media", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "image_path": args.image_path,
        "output_dir": args.output_dir,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "positional_dim": args.positional_dim,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "save_outputs": not args.no_save,
        "track_order_parameters": not args.no_order_tracking,
        "save_epoch_media": not args.no_epoch_media,
    }
    if args.variant_family == "hvk2d":
        outputs = run_hvk2d(**common)
    else:
        outputs = run_hvk1d(model_variant=args.model_variant, **common)
    final_loss = outputs["history"]["total_loss"][-1]
    print(f"HVK1D complete. Final loss: {final_loss:.6f}")
    if not args.no_save:
        print(f"Outputs saved to: {args.output_dir}")
