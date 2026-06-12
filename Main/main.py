import argparse
import json
from pathlib import Path

from src.training.training import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_DIR,
    load_config,
    resolve_path,
    train,
)


def build_run_config(config_path: str | Path | None = DEFAULT_CONFIG_PATH, **overrides):
    config = {
        "image_path": DEFAULT_IMAGE_PATH,
        "image_size": 256,
        "patch_size": 64,
        "positional_dim": 8,
        "steps": 120,
        "lr": 0.003,
        "device": "auto",
        "output_dir": DEFAULT_OUTPUT_DIR,
        "save_outputs": True,
        "show_plots": False,
        "track_order_parameters": True,
        "save_epoch_media": True,
        "epoch_frame_interval": 1,
        "zero_latent_uses_positions": False,
    }

    config.update(load_config(config_path))

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config["image_path"] = resolve_path(config["image_path"])
    config["output_dir"] = resolve_path(config["output_dir"])
    return config


def run_hvk_analysis(config_path: str | Path | None = DEFAULT_CONFIG_PATH, **overrides):
    config = build_run_config(config_path=config_path, **overrides)
    model, decoder, outputs = train(**config)
    return model, decoder, outputs, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Hamiltonian Vision Kernel analysis pipeline."
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
    parser.add_argument("--no-order-tracking", action="store_true")
    parser.add_argument("--no-epoch-media", action="store_true")
    parser.add_argument("--epoch-frame-interval", type=int)
    parser.add_argument("--zero-latent-uses-positions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "image_path": args.image_path,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "positional_dim": args.positional_dim,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "output_dir": args.output_dir,
        "save_outputs": False if args.no_save else None,
        "show_plots": True if args.show_plots else None,
        "track_order_parameters": False if args.no_order_tracking else None,
        "save_epoch_media": False if args.no_epoch_media else None,
        "epoch_frame_interval": args.epoch_frame_interval,
        "zero_latent_uses_positions": (
            True if args.zero_latent_uses_positions else None
        ),
    }

    _, _, outputs, config = run_hvk_analysis(
        config_path=args.config,
        **overrides,
    )

    summary = {
        "final_total_loss": outputs["history"]["total_loss"][-1],
        "final_reconstruction_loss": outputs["history"]["reconstruction_loss"][-1],
        "final_energy_loss": outputs["history"]["energy_loss"][-1],
        "phase_transition": outputs["phase_transition"],
        "phase_transition_order_parameter_gif": outputs.get("media", {}).get(
            "phase_transition_epoch_vs_order_parameter_gif"
        ),
        "phase_transition_merged_gif": outputs.get("media", {}).get(
            "phase_transition_order_parameter_reconstruction_gif"
        ),
        "output_dir": str(config["output_dir"]),
    }

    print("HVK analysis complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
