"""Stride-one overlapping-patch follow-up for the quantum thermal image model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_ablation_sweep import classical_baselines
from run_experiment import (
    DEFAULT_IMAGE,
    DEFAULT_OUTPUT,
    build_operators,
    load_image,
    make_completion_mask,
    metrics,
    save_gray,
    thermal_observables,
)
from scipy.stats import ttest_rel


def reconstruct_overlapping(
    image: np.ndarray,
    observed_mask: np.ndarray,
    temperatures: list[float],
    patch_shape: tuple[int, int],
    coupling: float,
    transverse_field: float,
    field_strength: float,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Measure every stride-one patch and average each pixel's measurements."""
    patch_h, patch_w = patch_shape
    z_ops, x_ops, bonds = build_operators(patch_h, patch_w)
    magnetization_sums = {temperature: np.zeros_like(image) for temperature in temperatures}
    energy_sums = {temperature: np.zeros_like(image) for temperature in temperatures}
    counts = np.zeros_like(image)

    for top in range(image.shape[0] - patch_h + 1):
        for left in range(image.shape[1] - patch_w + 1):
            patch = image[top : top + patch_h, left : left + patch_w]
            patch_mask = observed_mask[top : top + patch_h, left : left + patch_w]
            fields = field_strength * (2.0 * patch.reshape(-1) - 1.0) * patch_mask.reshape(-1)
            measured = thermal_observables(
                fields,
                temperatures,
                coupling,
                transverse_field,
                z_ops,
                x_ops,
                bonds,
            )
            counts[top : top + patch_h, left : left + patch_w] += 1.0
            for temperature, (z_expectations, local_energy) in measured.items():
                magnetization_sums[temperature][top : top + patch_h, left : left + patch_w] += (
                    z_expectations.reshape(patch_shape)
                )
                energy_sums[temperature][top : top + patch_h, left : left + patch_w] += local_energy.reshape(
                    patch_shape
                )

    reconstructions = {
        temperature: (1.0 + magnetization_sums[temperature] / counts) / 2.0 for temperature in temperatures
    }
    heat_maps = {temperature: energy_sums[temperature] / counts for temperature in temperatures}
    return reconstructions, heat_maps


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[float]] = {}
    for row in rows:
        groups.setdefault((row["Gamma"], row["temperature"]), []).append(row["psnr_db"])
    return [
        {
            "Gamma": gamma,
            "temperature": temperature,
            "n": len(values),
            "mean_psnr_db": float(np.mean(values)),
            "std_psnr_db": float(np.std(values, ddof=1)),
            "min_psnr_db": float(np.min(values)),
            "max_psnr_db": float(np.max(values)),
        }
        for (gamma, temperature), values in sorted(groups.items())
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, summary: list[dict], gaussian_values: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    for gamma in sorted({row["Gamma"] for row in summary}):
        points = [row for row in summary if row["Gamma"] == gamma]
        ax.errorbar(
            [row["temperature"] for row in points],
            [row["mean_psnr_db"] for row in points],
            yerr=[row["std_psnr_db"] for row in points],
            marker="o",
            capsize=3,
            label=f"Γ={gamma:g}" + (" (classical Ising)" if gamma == 0 else ""),
        )
    gaussian_mean = float(np.mean(gaussian_values))
    ax.axhline(gaussian_mean, color="black", linestyle=":", label=f"Gaussian baseline ({gaussian_mean:.2f} dB)")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Missing-pixel PSNR (dB)")
    ax.set_title("Stride-one overlapping quantum patches\nmean ± sample std over masks")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "overlapping_patches")
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--patch-height", type=int, default=2)
    parser.add_argument("--patch-width", type=int, default=3)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75])
    parser.add_argument("--coupling", type=float, default=0.7)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35])
    parser.add_argument("--field-strength", type=float, default=2.0)
    parser.add_argument("--observed-fraction", type=float, default=0.55)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 19, 29, 41])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image, (args.height, args.width))
    rows: list[dict] = []
    gaussian_values: list[float] = []
    saved_predictions: dict[tuple[int, float, float], np.ndarray] = {}

    for seed_index, seed in enumerate(args.seeds, start=1):
        print(f"Overlapping mask {seed_index}/{len(args.seeds)} (seed={seed})", flush=True)
        mask = make_completion_mask(image.shape, args.observed_fraction, seed)
        missing = ~mask
        gaussian_values.append(metrics(image, classical_baselines(image, mask)["gaussian"], missing)["psnr_db"])
        for gamma in args.gammas:
            reconstructions, _ = reconstruct_overlapping(
                image,
                mask,
                args.temperatures,
                (args.patch_height, args.patch_width),
                args.coupling,
                gamma,
                args.field_strength,
            )
            for temperature, prediction in reconstructions.items():
                result = metrics(image, prediction, missing)
                rows.append({"seed": seed, "Gamma": gamma, "temperature": temperature, **result})
                if seed == args.seeds[0]:
                    saved_predictions[(seed, gamma, temperature)] = prediction

    summary = aggregate(rows)
    best_classical = max((row for row in summary if row["Gamma"] == 0), key=lambda row: row["mean_psnr_db"])
    best_quantum = max((row for row in summary if row["Gamma"] > 0), key=lambda row: row["mean_psnr_db"])
    delta = best_quantum["mean_psnr_db"] - best_classical["mean_psnr_db"]
    classical_by_seed = {
        row["seed"]: row["psnr_db"]
        for row in rows
        if row["Gamma"] == best_classical["Gamma"] and row["temperature"] == best_classical["temperature"]
    }
    quantum_by_seed = {
        row["seed"]: row["psnr_db"]
        for row in rows
        if row["Gamma"] == best_quantum["Gamma"] and row["temperature"] == best_quantum["temperature"]
    }
    paired_deltas = np.array(
        [quantum_by_seed[seed] - classical_by_seed[seed] for seed in args.seeds], dtype=np.float64
    )
    paired_sem = float(np.std(paired_deltas, ddof=1) / np.sqrt(len(paired_deltas)))
    paired_test = ttest_rel(
        [quantum_by_seed[seed] for seed in args.seeds],
        [classical_by_seed[seed] for seed in args.seeds],
    )
    report = {
        "method": (
            "Stride-one overlapping six-qubit patches. Measurements for a pixel are averaged over every patch "
            "containing it. This removes hard tiling boundaries but is not a single globally coupled quantum state."
        ),
        "image_shape": list(image.shape),
        "patch_shape": [args.patch_height, args.patch_width],
        "patches_per_mask": (args.height - args.patch_height + 1) * (args.width - args.patch_width + 1),
        "seeds": args.seeds,
        "coupling_J": args.coupling,
        "temperatures": args.temperatures,
        "gammas": args.gammas,
        "best_classical_ising": best_classical,
        "best_quantum": best_quantum,
        "best_quantum_minus_best_classical_ising_psnr_db": delta,
        "paired_quantum_minus_classical": {
            "per_seed_psnr_db": paired_deltas.tolist(),
            "mean_psnr_db": float(np.mean(paired_deltas)),
            "std_psnr_db": float(np.std(paired_deltas, ddof=1)),
            "approx_95pct_ci_psnr_db": [
                float(np.mean(paired_deltas) - 1.96 * paired_sem),
                float(np.mean(paired_deltas) + 1.96 * paired_sem),
            ],
            "paired_ttest_pvalue_two_sided": float(paired_test.pvalue),
            "note": (
                "Exploratory test after selecting the best settings; "
                "it is not a preregistered confirmatory p-value."
            ),
        },
        "gaussian_baseline_mean_psnr_db": float(np.mean(gaussian_values)),
        "gaussian_baseline_std_psnr_db": float(np.std(gaussian_values, ddof=1)),
    }
    write_csv(args.output / "all_runs.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    save_plot(args.output / "overlapping_ablation.png", summary, gaussian_values)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    example_seed = args.seeds[0]
    save_gray(args.output / "target.png", image)
    save_gray(
        args.output / "best_classical_example.png",
        saved_predictions[(example_seed, best_classical["Gamma"], best_classical["temperature"])],
    )
    save_gray(
        args.output / "best_quantum_example.png",
        saved_predictions[(example_seed, best_quantum["Gamma"], best_quantum["temperature"])],
    )
    print(json.dumps(report, indent=2))
    print(f"Saved overlapping-patch results to: {args.output}")


if __name__ == "__main__":
    main()
