"""Extends supplementary_study.tex's Table tab:hamiltonian_reproducibility with the
two Table tab:hamiltonian_controls sector variants that fresh reproduction never
covered: "No observable noise" and "ZZ-only observables" (see TODO/results-core-map.md
row R10). The other 3 original rows (legacy signed energy, no energy loss, contrastive)
were already fresh-reproduced -- see Main_new/README.md and
supplementary_study.tex Section "A Reproducibility Note on Table hamiltonian_controls".

Same protocol as that existing reproduction, for direct comparability: Monalisa,
256x256 image / patch=64 (training_config.json defaults), model_variant=standard,
200 steps, seed=42, energy_loss_mode=linear (Table hamiltonian_controls's original
"legacy baseline" objective -- NOT Main_new's new default "positive" mode, since these
two new rows are meant to sit in the fresh-but-legacy-objective reproduction table
alongside the other three, not the bounded-coupling-fix rows below it).

no-obs-noise sets observable_noise=False; zz-only sets observable_set="zz-only" --
both via Main_new/src/training/training.py's existing `ablation_mode` kwarg (see that
file's `valid_ablation_modes`), no new ablation code written here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_NEW_DIR = ROOT / "Main_new"
if str(MAIN_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_NEW_DIR))

from src.training.training import train  # noqa: E402

RESULT_DIR = ROOT / "Main2" / "newHVK" / "results" / "hamiltonian_reproducibility_extension"

STEPS = 200
SEED = 42

# (label matching tab:hamiltonian_controls's row name, ablation_mode)
VARIANTS = [
    ("No observable noise", "no-obs-noise"),
    ("ZZ-only observables", "zz-only"),
]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, ablation_mode in VARIANTS:
        print(f"Running {label} (ablation_mode={ablation_mode})...", flush=True)
        _, _, outputs = train(
            steps=STEPS,
            ablation_mode=ablation_mode,
            energy_loss_mode="linear",
            model_variant="standard",
            seed=SEED,
            save_outputs=False,
            track_order_parameters=False,
            save_epoch_media=False,
            show_plots=False,
            device="cpu",
        )
        metrics = outputs["reconstruction_metrics"]
        row = {
            "label": label,
            "ablation_mode": ablation_mode,
            "energy_loss_mode": "linear",
            "steps": STEPS,
            "seed": SEED,
            "psnr_db": metrics["psnr"],
            "ssim": metrics["ssim"],
            "mse": metrics["mse"],
            "note": (
                "Table hamiltonian_controls's original row, rerun"
                if label != "unused"
                else ""
            ),
        }
        rows.append(row)
        print(f"  -> PSNR={row['psnr_db']:.2f} dB SSIM={row['ssim']:.4f}", flush=True)

    output_path = RESULT_DIR / "summary.json"
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
