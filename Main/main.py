import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
        "model_variant": "both",
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
    if config.get("model_variant") == "both":
        run_outputs = {}
        models = {}
        decoders = {}
        base_output_dir = Path(config["output_dir"])
        for variant, folder in [
            ("standard", "standard_hvk1d"),
            ("symmetric", "symmetric_hvk1d"),
        ]:
            label = "Standard HVK1D" if variant == "standard" else "Symmetric HVK1D"
            print(f"\n=== Running {label} ===")
            variant_config = dict(config)
            variant_config["model_variant"] = variant
            variant_config["output_dir"] = base_output_dir / folder
            variant_config["log_prefix"] = label
            model, decoder, outputs = train(**variant_config)
            models[variant] = model
            decoders[variant] = decoder
            run_outputs[variant] = outputs
            print(
                f"=== Finished {label}: "
                f"final_loss={outputs['history']['total_loss'][-1]:.6f}, "
                f"output_dir={variant_config['output_dir']} ==="
            )
        comparison_path = (
            save_variant_comparison(run_outputs, base_output_dir)
            if config.get("save_outputs", True)
            else None
        )
        outputs = {
            "variants": run_outputs,
            "media": (
                {"variant_comparison": str(comparison_path)}
                if comparison_path is not None
                else {}
            ),
            "history": run_outputs["standard"]["history"],
            "phase_transition": {
                "standard": run_outputs["standard"]["phase_transition"],
                "symmetric": run_outputs["symmetric"]["phase_transition"],
            },
        }
        return models, decoders, outputs, config

    if "log_prefix" not in config:
        config["log_prefix"] = (
            "Symmetric HVK1D"
            if config.get("model_variant") == "symmetric"
            else "Standard HVK1D"
        )
    print(f"\n=== Running {config['log_prefix']} ===")
    model, decoder, outputs = train(**config)
    print(
        f"=== Finished {config['log_prefix']}: "
        f"final_loss={outputs['history']['total_loss'][-1]:.6f}, "
        f"output_dir={config['output_dir']} ==="
    )
    return model, decoder, outputs, config


def save_variant_comparison(outputs_by_variant: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hvk1d_standard_vs_symmetric.png"

    standard = outputs_by_variant["standard"]
    symmetric = outputs_by_variant["symmetric"]
    panels = [
        ("Original", standard["original"]),
        ("Standard HVK1D", standard["quantum_reconstruction"]),
        ("Symmetric HVK1D", symmetric["quantum_reconstruction"]),
        (
            "Absolute Difference",
            np.abs(
                standard["quantum_reconstruction"]
                - symmetric["quantum_reconstruction"]
            ),
        ),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.5))
    for ax, (title, image) in zip(axes, panels):
        cmap = "magma" if title == "Absolute Difference" else "gray"
        ax.imshow(np.clip(image, 0, 1), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("HVK1D Standard vs U(1)-Symmetric Reconstruction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    metrics = {
        variant: {
            "final_total_loss": values["history"]["total_loss"][-1],
            "final_reconstruction_loss": values["history"]["reconstruction_loss"][-1],
            "final_energy_loss": values["history"]["energy_loss"][-1],
            "phase_transition": values["phase_transition"],
            "media": values.get("media", {}),
        }
        for variant, values in outputs_by_variant.items()
    }
    metrics["comparison_plot"] = str(output_path)
    (output_dir / "hvk1d_standard_vs_symmetric_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return output_path


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
    parser.add_argument("--shuffle-observables-at-eval", action="store_true")
    parser.add_argument(
        "--model-variant",
        choices=["standard", "symmetric", "both"],
    )
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
        "shuffle_observables_at_eval": (
            True if args.shuffle_observables_at_eval else None
        ),
        "model_variant": args.model_variant,
    }

    _, _, outputs, config = run_hvk_analysis(
        config_path=args.config,
        **overrides,
    )

    if config.get("model_variant") == "both":
        summary = {
            "model_variant": "both",
            "standard_final_loss": outputs["variants"]["standard"]["history"][
                "total_loss"
            ][-1],
            "symmetric_final_loss": outputs["variants"]["symmetric"]["history"][
                "total_loss"
            ][-1],
            "phase_transition": outputs["phase_transition"],
            "comparison_plot": outputs.get("media", {}).get("variant_comparison"),
            "output_dir": str(config["output_dir"]),
        }
    else:
        summary = {
            "model_variant": config.get("model_variant", "standard"),
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


# CLI entry point (also exposed as hvk-run)
def cli_entry() -> None:
    main()


if __name__ == "__main__":
    main()
