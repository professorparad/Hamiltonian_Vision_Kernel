from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MAIN_DIR = REPO_ROOT / "Main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from Main2.src.config import Main2Config
from Main2.src.training import run_main2


def parse_args():
    defaults = Main2Config()
    parser = argparse.ArgumentParser(
        description="Run the modular Main2 HVK 2D grid experiment."
    )
    parser.add_argument("--image-path", type=Path, default=defaults.image_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--patch-size", type=int, default=defaults.patch_size)
    parser.add_argument("--positional-dim", type=int, default=defaults.positional_dim)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=defaults.device)
    parser.add_argument("--frame-interval", type=int, default=defaults.frame_interval)
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Main2Config(
        image_path=args.image_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
        positional_dim=args.positional_dim,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        frame_interval=args.frame_interval,
        save_gif=not args.no_gif,
    )
    result = run_main2(config)
    print("Main2 HVK run complete.")
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "epochs_recorded": len(result["epoch_rows"]),
                "gif_path": str(result["gif_path"]) if result["gif_path"] else None,
                "order_gif_path": (
                    str(result["order_gif_path"])
                    if result["order_gif_path"]
                    else None
                ),
                "merged_gif_path": (
                    str(result["merged_gif_path"])
                    if result["merged_gif_path"]
                    else None
                ),
                "phase_transition": result["phase_transition"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
