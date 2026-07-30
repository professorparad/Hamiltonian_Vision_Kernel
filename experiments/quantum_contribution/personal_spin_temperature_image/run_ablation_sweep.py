"""Repeated-mask ablation sweep for the private quantum thermal image study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_experiment import (
    DEFAULT_IMAGE,
    DEFAULT_OUTPUT,
    load_image,
    make_completion_mask,
    metrics,
    reconstruct,
)
from scipy.ndimage import distance_transform_edt, gaussian_filter


def classical_baselines(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    observed = image[mask]
    mean_fill = np.where(mask, image, float(observed.mean()))

    # For every missing pixel, select the closest observed pixel.
    _, nearest_indices = distance_transform_edt(~mask, return_indices=True)
    nearest = image[tuple(nearest_indices)]
    nearest = np.where(mask, image, nearest)

    # Normalized convolution prevents missing values from being treated as zeros.
    weighted_values = gaussian_filter(image * mask, sigma=1.0, mode="nearest")
    weights = gaussian_filter(mask.astype(np.float64), sigma=1.0, mode="nearest")
    gaussian = np.where(mask, image, weighted_values / np.maximum(weights, 1e-12))
    return {"mean_fill": mean_fill, "nearest": nearest, "gaussian": gaussian}


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["family"], row["label"], row["temperature"], row["J"], row["Gamma"])
        groups.setdefault(key, []).append(row["psnr_db"])
    summary = []
    for key, values in groups.items():
        family, label, temperature, coupling, gamma = key
        summary.append(
            {
                "family": family,
                "label": label,
                "temperature": temperature,
                "J": coupling,
                "Gamma": gamma,
                "n": len(values),
                "mean_psnr_db": float(np.mean(values)),
                "std_psnr_db": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min_psnr_db": float(np.min(values)),
                "max_psnr_db": float(np.max(values)),
            }
        )
    return summary


def best_row(rows: list[dict], family: str) -> dict:
    eligible = [row for row in rows if row["family"] == family]
    return max(eligible, key=lambda row: row["mean_psnr_db"])


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output: Path, summary: list[dict], temperatures: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    configurations = sorted(
        {(row["J"], row["Gamma"]) for row in summary if row["family"] in {"classical_ising", "quantum"}},
        key=lambda item: (item[1], item[0]),
    )
    for coupling, gamma in configurations:
        points = sorted(
            [
                row
                for row in summary
                if row["J"] == coupling
                and row["Gamma"] == gamma
                and row["family"] in {"classical_ising", "quantum"}
            ],
            key=lambda row: row["temperature"],
        )
        style = "--" if gamma == 0 else "-"
        axes[0].plot(
            [row["temperature"] for row in points],
            [row["mean_psnr_db"] for row in points],
            linestyle=style,
            marker="o",
            markersize=3,
            label=f"J={coupling:g}, Γ={gamma:g}",
        )
    axes[0].set_xlabel("Temperature T")
    axes[0].set_ylabel("Missing-pixel PSNR (dB)")
    axes[0].set_title("Thermal reconstruction sweep")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    contenders = []
    for family in ("baseline", "local_only", "classical_ising", "quantum"):
        family_rows = [row for row in summary if row["family"] == family]
        if family_rows:
            contenders.append(max(family_rows, key=lambda row: row["mean_psnr_db"]))
    labels = [row["label"] for row in contenders]
    means = [row["mean_psnr_db"] for row in contenders]
    errors = [row["std_psnr_db"] for row in contenders]
    colors = ["#777777", "#d18f00", "#3b75af", "#8a4fa3"][: len(contenders)]
    axes[1].bar(labels, means, yerr=errors, capsize=4, color=colors)
    axes[1].set_ylabel("Best mean missing-pixel PSNR (dB)")
    axes[1].set_title(f"Best configuration by family ({len(temperatures)} temperatures)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Quantum thermal image completion: repeated-mask ablation")
    fig.savefig(output / "ablation_summary.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "ablation_sweep")
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--patch-height", type=int, default=2)
    parser.add_argument("--patch-width", type=int, default=3)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--couplings", type=float, nargs="+", default=[0.25, 0.45, 0.7])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.2, 0.35])
    parser.add_argument("--field-strength", type=float, default=2.0)
    parser.add_argument("--observed-fraction", type=float, default=0.55)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 19, 29, 41])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image, (args.height, args.width))
    rows: list[dict] = []

    for seed_index, seed in enumerate(args.seeds, start=1):
        print(f"Mask {seed_index}/{len(args.seeds)} (seed={seed})", flush=True)
        mask = make_completion_mask(image.shape, args.observed_fraction, seed)
        missing = ~mask

        for label, prediction in classical_baselines(image, mask).items():
            result = metrics(image, prediction, missing)
            rows.append(
                {
                    "seed": seed,
                    "family": "baseline",
                    "label": label,
                    "temperature": "",
                    "J": "",
                    "Gamma": "",
                    **result,
                }
            )

        for coupling in args.couplings:
            for gamma in args.gammas:
                reconstructions, _ = reconstruct(
                    image,
                    mask,
                    args.temperatures,
                    (args.patch_height, args.patch_width),
                    coupling,
                    gamma,
                    args.field_strength,
                )
                if coupling == 0:
                    family = "local_only"
                elif gamma == 0:
                    family = "classical_ising"
                else:
                    family = "quantum"
                for temperature, prediction in reconstructions.items():
                    result = metrics(image, prediction, missing)
                    rows.append(
                        {
                            "seed": seed,
                            "family": family,
                            "label": f"{family}: J={coupling:g}, Gamma={gamma:g}",
                            "temperature": temperature,
                            "J": coupling,
                            "Gamma": gamma,
                            **result,
                        }
                    )

        # Explicit no-neighbour quantum control.
        for gamma in (0.0, 0.35):
            reconstructions, _ = reconstruct(
                image,
                mask,
                args.temperatures,
                (args.patch_height, args.patch_width),
                0.0,
                gamma,
                args.field_strength,
            )
            for temperature, prediction in reconstructions.items():
                result = metrics(image, prediction, missing)
                rows.append(
                    {
                        "seed": seed,
                        "family": "local_only",
                        "label": f"local_only: J=0, Gamma={gamma:g}",
                        "temperature": temperature,
                        "J": 0.0,
                        "Gamma": gamma,
                        **result,
                    }
                )

    summary = summarize(rows)
    best = {family: best_row(summary, family) for family in ("baseline", "local_only", "classical_ising", "quantum")}
    quantum_delta = best["quantum"]["mean_psnr_db"] - best["classical_ising"]["mean_psnr_db"]
    report = {
        "interpretation": (
            "Quantum contribution is assessed as the best repeated-mask transverse-field result minus the "
            "best Gamma=0 classical-Ising result. A positive value is only simulation evidence, not quantum advantage."
        ),
        "image_shape": list(image.shape),
        "patch_shape": [args.patch_height, args.patch_width],
        "seeds": args.seeds,
        "observed_fraction": args.observed_fraction,
        "temperatures": args.temperatures,
        "couplings": args.couplings,
        "gammas": args.gammas,
        "best_by_family": best,
        "best_quantum_minus_best_classical_ising_psnr_db": quantum_delta,
    }

    save_csv(args.output / "all_runs.csv", rows)
    save_csv(args.output / "summary.csv", summary)
    save_plots(args.output, summary, args.temperatures)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved ablation results to: {args.output}")


if __name__ == "__main__":
    main()
