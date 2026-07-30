"""Frozen-parameter held-out validation of the overlapping quantum spin model.

The configuration and analysis contract are written before any held-out mask is
evaluated. No parameter selection is performed in this script.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_ablation_sweep import classical_baselines
from run_experiment import DEFAULT_IMAGE, DEFAULT_OUTPUT, load_image, make_completion_mask, metrics
from run_overlapping_patches import reconstruct_overlapping
from scipy.stats import ttest_rel, wilcoxon

OUTPUT = DEFAULT_OUTPUT / "preregistered_validation"
PREREGISTRATION_PATH = OUTPUT / "preregistration.json"
CHECKPOINT_PATH = OUTPUT / "heldout_runs.csv"
REPORT_PATH = OUTPUT / "report.json"

# Frozen from the exploratory overlapping-patch sweep.
IMAGE_SHAPE = (24, 24)
PATCH_SHAPE = (2, 3)
OBSERVED_FRACTION = 0.55
FIELD_STRENGTH = 2.0
COUPLING_J = 0.7
TEMPERATURE = 1.25
CLASSICAL_GAMMA = 0.0
QUANTUM_GAMMA = 0.35

# These masks were not used in the exploratory runs.
HELDOUT_SEEDS = [
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
]


def preregistration() -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "parameters frozen before held-out mask evaluation",
        "primary_hypothesis": (
            "At fixed J=0.7 and T=1.25, the Gamma=0.35 transverse-field model has a positive mean paired "
            "missing-pixel PSNR difference relative to the Gamma=0 classical-Ising control."
        ),
        "primary_endpoint": "paired PSNR difference in dB: quantum minus classical Ising",
        "primary_test": "two-sided paired t-test at alpha=0.05",
        "estimation": "mean paired difference and percentile bootstrap 95% confidence interval",
        "secondary_test": "two-sided Wilcoxon signed-rank test",
        "decision_rule": (
            "Call the held-out result statistically positive only if mean delta > 0, the bootstrap 95% interval "
            "excludes zero, and the two-sided paired t-test p-value is below 0.05."
        ),
        "no_tuning_rule": (
            "No J, T, Gamma, patch shape, field strength, or mask seed may be "
            "changed after evaluation starts."
        ),
        "configuration": {
            "image": str(DEFAULT_IMAGE),
            "image_shape": list(IMAGE_SHAPE),
            "patch_shape": list(PATCH_SHAPE),
            "stride": 1,
            "observed_fraction": OBSERVED_FRACTION,
            "field_strength": FIELD_STRENGTH,
            "coupling_J": COUPLING_J,
            "temperature": TEMPERATURE,
            "classical_Gamma": CLASSICAL_GAMMA,
            "quantum_Gamma": QUANTUM_GAMMA,
            "heldout_seeds": HELDOUT_SEEDS,
        },
        "interpretation_limit": (
            "A positive result is evidence for a transverse-field contribution within this exact quantum-model "
            "simulation. It is not evidence of computational quantum advantage or quantum-hardware performance."
        ),
    }


def write_rows(rows: list[dict]) -> None:
    with CHECKPOINT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(deltas: np.ndarray, seed: int = 20260724, samples: int = 50_000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    bootstrap_means = deltas[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return [float(low), float(high)]


def save_validation_plot(path: Path, classical: np.ndarray, quantum: np.ndarray, gaussian: np.ndarray) -> None:
    deltas = quantum - classical
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for index, (classical_value, quantum_value) in enumerate(zip(classical, quantum, strict=True)):
        axes[0].plot([0, 1], [classical_value, quantum_value], color="#888888", alpha=0.55, linewidth=0.8)
        axes[0].scatter([0, 1], [classical_value, quantum_value], color=["#3b75af", "#8a4fa3"], s=18)
    axes[0].set_xticks([0, 1], ["Γ=0\nclassical Ising", "Γ=0.35\nquantum"])
    axes[0].set_ylabel("Missing-pixel PSNR (dB)")
    axes[0].set_title("Paired held-out masks")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist(deltas, bins=10, color="#8a4fa3", alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(float(deltas.mean()), color="#b51f1f", linewidth=2, label=f"mean = {deltas.mean():.3f} dB")
    axes[1].set_xlabel("Quantum − classical Ising PSNR (dB)")
    axes[1].set_ylabel("Held-out masks")
    axes[1].set_title(f"All {len(deltas)} paired differences")
    axes[1].legend()
    fig.suptitle(
        f"Frozen held-out validation; Gaussian baseline mean = {gaussian.mean():.2f} dB",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise RuntimeError(f"Validation already completed; refusing to overwrite: {REPORT_PATH}")
    if not PREREGISTRATION_PATH.exists():
        PREREGISTRATION_PATH.write_text(json.dumps(preregistration(), indent=2), encoding="utf-8")

    image = load_image(DEFAULT_IMAGE, IMAGE_SHAPE)
    rows: list[dict] = []
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {int(row["seed"]) for row in rows}
    else:
        completed = set()

    for index, seed in enumerate(HELDOUT_SEEDS, start=1):
        if seed in completed:
            print(f"Held-out mask {index}/{len(HELDOUT_SEEDS)} seed={seed} already complete", flush=True)
            continue
        print(f"Held-out mask {index}/{len(HELDOUT_SEEDS)} seed={seed}", flush=True)
        mask = make_completion_mask(image.shape, OBSERVED_FRACTION, seed)
        missing = ~mask
        predictions: dict[float, np.ndarray] = {}
        for gamma in (CLASSICAL_GAMMA, QUANTUM_GAMMA):
            reconstructions, _ = reconstruct_overlapping(
                image,
                mask,
                [TEMPERATURE],
                PATCH_SHAPE,
                COUPLING_J,
                gamma,
                FIELD_STRENGTH,
            )
            predictions[gamma] = reconstructions[TEMPERATURE]
        classical_psnr = metrics(image, predictions[CLASSICAL_GAMMA], missing)["psnr_db"]
        quantum_psnr = metrics(image, predictions[QUANTUM_GAMMA], missing)["psnr_db"]
        gaussian_psnr = metrics(image, classical_baselines(image, mask)["gaussian"], missing)["psnr_db"]
        rows.append(
            {
                "seed": seed,
                "actual_observed_fraction": float(mask.mean()),
                "classical_ising_psnr_db": classical_psnr,
                "quantum_psnr_db": quantum_psnr,
                "quantum_minus_classical_db": quantum_psnr - classical_psnr,
                "gaussian_psnr_db": gaussian_psnr,
            }
        )
        write_rows(rows)

    # Re-read numeric data so resumed and uninterrupted runs follow one path.
    with CHECKPOINT_PATH.open(newline="", encoding="utf-8") as handle:
        final_rows = list(csv.DictReader(handle))
    classical = np.array([float(row["classical_ising_psnr_db"]) for row in final_rows])
    quantum = np.array([float(row["quantum_psnr_db"]) for row in final_rows])
    gaussian = np.array([float(row["gaussian_psnr_db"]) for row in final_rows])
    deltas = quantum - classical
    paired_t = ttest_rel(quantum, classical)
    signed_rank = wilcoxon(deltas, alternative="two-sided")
    ci = bootstrap_ci(deltas)
    positive = bool(float(deltas.mean()) > 0 and ci[0] > 0 and float(paired_t.pvalue) < 0.05)
    save_validation_plot(OUTPUT / "heldout_validation.png", classical, quantum, gaussian)

    report = {
        "preregistration": str(PREREGISTRATION_PATH),
        "n_heldout_masks": len(final_rows),
        "frozen_configuration": preregistration()["configuration"],
        "classical_ising": {
            "mean_psnr_db": float(classical.mean()),
            "std_psnr_db": float(classical.std(ddof=1)),
        },
        "quantum": {
            "mean_psnr_db": float(quantum.mean()),
            "std_psnr_db": float(quantum.std(ddof=1)),
        },
        "paired_quantum_minus_classical": {
            "mean_psnr_db": float(deltas.mean()),
            "std_psnr_db": float(deltas.std(ddof=1)),
            "median_psnr_db": float(np.median(deltas)),
            "bootstrap_95pct_ci_psnr_db": ci,
            "positive_masks": int(np.sum(deltas > 0)),
            "negative_masks": int(np.sum(deltas < 0)),
            "paired_ttest_statistic": float(paired_t.statistic),
            "paired_ttest_pvalue_two_sided": float(paired_t.pvalue),
            "wilcoxon_statistic": float(signed_rank.statistic),
            "wilcoxon_pvalue_two_sided": float(signed_rank.pvalue),
        },
        "gaussian_baseline": {
            "mean_psnr_db": float(gaussian.mean()),
            "std_psnr_db": float(gaussian.std(ddof=1)),
            "quantum_minus_gaussian_mean_psnr_db": float((quantum - gaussian).mean()),
        },
        "preregistered_positive_result": positive,
        "interpretation": (
            "This held-out test isolates the contribution of Gamma within an exact simulation of small quantum "
            "Gibbs systems. It does not establish computational quantum advantage."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved frozen held-out validation to: {OUTPUT}")


if __name__ == "__main__":
    main()
