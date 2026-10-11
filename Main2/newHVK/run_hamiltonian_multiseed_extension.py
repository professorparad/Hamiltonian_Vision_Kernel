"""Run the missing seeds for the Hamiltonian-objective comparison.

This extends the fresh 200-step, seed-42 comparison documented in
``Main_new/README.md``.  It intentionally imports the unmodified ``Main`` model so
that the signed and contrastive rows retain the original unbounded couplings.

Each completed run is written immediately, making the campaign safe to resume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "Main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from src.training.training import train  # noqa: E402


RESULT_DIR = (
    ROOT / "Main2" / "newHVK" / "results" / "hamiltonian_multiseed_extension"
)
OUTPUT_PATH = RESULT_DIR / "runs.json"
STEPS = 200
SEEDS = (43, 44)
VARIANTS = (
    ("Signed energy", "linear"),
    ("Contrastive energy", "contrastive"),
)


def load_rows() -> list[dict]:
    if not OUTPUT_PATH.exists():
        return []
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def save_rows(rows: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    completed = {(row["energy_loss_mode"], row["seed"]) for row in rows}

    for label, energy_loss_mode in VARIANTS:
        for seed in SEEDS:
            key = (energy_loss_mode, seed)
            if key in completed:
                print(f"Skipping completed run: {label}, seed {seed}", flush=True)
                continue

            print(f"Running {label}, seed {seed}...", flush=True)
            _, _, outputs = train(
                steps=STEPS,
                energy_loss_mode=energy_loss_mode,
                model_variant="standard",
                seed=seed,
                save_outputs=False,
                track_order_parameters=False,
                save_epoch_media=False,
                show_plots=False,
                device="cpu",
            )
            metrics = outputs["reconstruction_metrics"]
            row = {
                "label": label,
                "energy_loss_mode": energy_loss_mode,
                "steps": STEPS,
                "seed": seed,
                "psnr_db": metrics["psnr"],
                "ssim": metrics["ssim"],
                "mse": metrics["mse"],
            }
            rows.append(row)
            rows.sort(key=lambda item: (item["energy_loss_mode"], item["seed"]))
            save_rows(rows)
            completed.add(key)
            print(
                f"  -> PSNR={row['psnr_db']:.2f} dB "
                f"SSIM={row['ssim']:.4f}",
                flush=True,
            )

    print(f"Saved {len(rows)} runs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
