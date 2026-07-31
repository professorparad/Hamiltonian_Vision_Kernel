"""Prototype Thermodynamic Qubit Collector–Modulator Reconstruction (TQ-CMR)."""

from __future__ import annotations

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
    thermal_observables,
)
from run_overlapping_patches import reconstruct_overlapping
from scipy.stats import ttest_rel

OUTPUT = DEFAULT_OUTPUT / "collector_modulator_v2"
IMAGE_SHAPE = (24, 24)
PATCH_SHAPE = (2, 3)
SEEDS = [3, 7, 11, 19, 29, 41]  # exploratory masks only
OBSERVED_FRACTION = 0.55
FIELD_STRENGTH = 2.0

COLLECTOR_J = 0.7
COLLECTOR_GAMMA = 0.35
COLLECTOR_TEMPERATURES = [1.0, 1.5]

MODULATOR_J = 0.7
MODULATOR_TEMPERATURE = 1.25
MODULATOR_GAMMAS = [0.0, 0.35]
# Revision 2: direct magnetization feedback caused self-reinforcing contrast
# collapse in revision 1. The collector now modulates interactions rather than
# injecting its estimate as a target-like longitudinal field.
FEEDBACK_STRENGTH = 0.0
EDGE_SENSITIVITY = 0.5
COUPLING_FLOOR = 0.2
TRANSVERSE_FLOOR = 0.15


def generalized_thermal_observables(
    fields: np.ndarray,
    temperature: float,
    bond_couplings: dict[tuple[int, int], float],
    transverse_fields: np.ndarray,
    z_ops: list[np.ndarray],
    x_ops: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    dimension = z_ops[0].shape[0]
    hamiltonian = np.zeros((dimension, dimension), dtype=np.float64)
    zz_ops: dict[tuple[int, int], np.ndarray] = {}
    for (i, j), coupling in bond_couplings.items():
        zz_ops[(i, j)] = z_ops[i] @ z_ops[j]
        hamiltonian -= coupling * zz_ops[(i, j)]
    for i in range(len(fields)):
        hamiltonian -= fields[i] * z_ops[i]
        hamiltonian -= transverse_fields[i] * x_ops[i]

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    weights = np.exp(-(eigenvalues - eigenvalues.min()) / temperature)
    probabilities = weights / weights.sum()

    def expected(operator: np.ndarray) -> float:
        diagonal = np.sum(eigenvectors * (operator @ eigenvectors), axis=0)
        return float(probabilities @ diagonal)

    magnetizations = np.array([expected(operator) for operator in z_ops])
    local_energy = -fields * magnetizations
    for i, operator in enumerate(x_ops):
        local_energy[i] -= transverse_fields[i] * expected(operator)
    for (i, j), coupling in bond_couplings.items():
        bond_energy = -coupling * expected(zz_ops[(i, j)])
        local_energy[i] += 0.5 * bond_energy
        local_energy[j] += 0.5 * bond_energy
    return magnetizations, local_energy


def collector_modulator_reconstruct(
    image: np.ndarray,
    mask: np.ndarray,
    modulator_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_h, patch_w = PATCH_SHAPE
    z_ops, x_ops, bonds = build_operators(patch_h, patch_w)
    output_sum = np.zeros_like(image)
    collector_sum = np.zeros_like(image)
    response_sum = np.zeros_like(image)
    energy_sum = np.zeros_like(image)
    counts = np.zeros_like(image)

    for top in range(image.shape[0] - patch_h + 1):
        for left in range(image.shape[1] - patch_w + 1):
            patch = image[top : top + patch_h, left : left + patch_w]
            patch_mask = mask[top : top + patch_h, left : left + patch_w]
            raw_fields = FIELD_STRENGTH * (2.0 * patch.reshape(-1) - 1.0) * patch_mask.reshape(-1)

            collected = thermal_observables(
                raw_fields,
                COLLECTOR_TEMPERATURES,
                COLLECTOR_J,
                COLLECTOR_GAMMA,
                z_ops,
                x_ops,
                bonds,
            )
            low_m = collected[COLLECTOR_TEMPERATURES[0]][0]
            high_m = collected[COLLECTOR_TEMPERATURES[1]][0]
            collector_m = 0.5 * (low_m + high_m)
            thermal_response = np.abs(low_m - high_m) / (
                COLLECTOR_TEMPERATURES[1] - COLLECTOR_TEMPERATURES[0]
            )
            confidence = np.abs(collector_m)

            modulated_fields = raw_fields.copy()
            missing = ~patch_mask.reshape(-1)
            modulated_fields[missing] += FEEDBACK_STRENGTH * collector_m[missing]
            bond_couplings = {
                (i, j): MODULATOR_J
                * (
                    COUPLING_FLOOR
                    + (1.0 - COUPLING_FLOOR)
                    * np.exp(-EDGE_SENSITIVITY * abs(collector_m[i] - collector_m[j]))
                )
                for i, j in bonds
            }
            transverse_fields = modulator_gamma * (
                TRANSVERSE_FLOOR + (1.0 - TRANSVERSE_FLOOR) * (1.0 - confidence)
            )
            modulator_m, local_energy = generalized_thermal_observables(
                modulated_fields,
                MODULATOR_TEMPERATURE,
                bond_couplings,
                transverse_fields,
                z_ops,
                x_ops,
            )

            window = np.s_[top : top + patch_h, left : left + patch_w]
            counts[window] += 1
            output_sum[window] += modulator_m.reshape(PATCH_SHAPE)
            collector_sum[window] += collector_m.reshape(PATCH_SHAPE)
            response_sum[window] += thermal_response.reshape(PATCH_SHAPE)
            energy_sum[window] += local_energy.reshape(PATCH_SHAPE)

    reconstruction = (1.0 + output_sum / counts) / 2.0
    collector_map = collector_sum / counts
    diagnostic_map = response_sum / counts
    heat_map = energy_sum / counts
    # Combine response and heat only in saved diagnostics, not reconstruction.
    return reconstruction, collector_map, diagnostic_map + 0.0 * heat_map


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(
    path: Path,
    rows: list[dict],
    target: np.ndarray,
    example: dict[str, np.ndarray],
) -> None:
    methods = ["one_pass_quantum", "cm_classical", "cm_quantum", "gaussian"]
    labels = ["One-pass\nquantum", "C–M\nΓ=0", "C–M\nΓ=0.35", "Gaussian"]
    means = [np.mean([row[name] for row in rows]) for name in methods]
    stds = [np.std([row[name] for row in rows], ddof=1) for name in methods]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.5), constrained_layout=True)
    axes[0, 0].bar(labels, means, yerr=stds, color=["#5479a8", "#5b9a72", "#8a4fa3", "#777777"], capsize=4)
    axes[0, 0].set_ylabel("Missing-pixel PSNR (dB)")
    axes[0, 0].set_title("Exploratory masks")
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Target")
    axes[0, 2].imshow(example["cm_classical"], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("C–M Γ=0")
    axes[0, 3].imshow(example["cm_quantum"], cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title("C–M Γ=0.35")
    axes[1, 0].imshow(example["observed"], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Observed input")
    axes[1, 1].imshow(example["collector"], cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_title("Collector magnetization")
    response_limit = float(np.max(np.abs(example["response"])))
    axes[1, 2].imshow(example["response"], cmap="magma", vmin=0, vmax=response_limit)
    axes[1, 2].set_title("Thermal response")
    delta = example["cm_quantum"] - example["cm_classical"]
    delta_limit = max(float(np.max(np.abs(delta))), 1e-8)
    axes[1, 3].imshow(delta, cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit)
    axes[1, 3].set_title("Quantum − classical pixels")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Thermodynamic Qubit Collector–Modulator Reconstruction", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = load_image(DEFAULT_IMAGE, IMAGE_SHAPE)
    rows: list[dict] = []
    example: dict[str, np.ndarray] = {}

    for index, seed in enumerate(SEEDS, start=1):
        print(f"Collector–modulator mask {index}/{len(SEEDS)} seed={seed}", flush=True)
        mask = make_completion_mask(image.shape, OBSERVED_FRACTION, seed)
        missing = ~mask
        one_pass, _ = reconstruct_overlapping(
            image,
            mask,
            [MODULATOR_TEMPERATURE],
            PATCH_SHAPE,
            MODULATOR_J,
            0.35,
            FIELD_STRENGTH,
        )
        cm_classical, collector, response = collector_modulator_reconstruct(image, mask, 0.0)
        cm_quantum, _, _ = collector_modulator_reconstruct(image, mask, 0.35)
        gaussian = classical_baselines(image, mask)["gaussian"]
        row = {"seed": seed}
        for name, prediction in {
            "one_pass_quantum": one_pass[MODULATOR_TEMPERATURE],
            "cm_classical": cm_classical,
            "cm_quantum": cm_quantum,
            "gaussian": gaussian,
        }.items():
            row[name] = metrics(image, prediction, missing)["psnr_db"]
        rows.append(row)
        if index == 1:
            example = {
                "observed": np.where(mask, image, 0.5),
                "cm_classical": cm_classical,
                "cm_quantum": cm_quantum,
                "collector": collector,
                "response": response,
            }

    cm_classical = np.array([row["cm_classical"] for row in rows])
    cm_quantum = np.array([row["cm_quantum"] for row in rows])
    one_pass = np.array([row["one_pass_quantum"] for row in rows])
    gaussian = np.array([row["gaussian"] for row in rows])
    quantum_test = ttest_rel(cm_quantum, cm_classical)
    modulation_test = ttest_rel(cm_quantum, one_pass)
    report = {
        "status": "exploratory prototype; modulation parameters are not held-out",
        "architecture": "Thermodynamic Qubit Collector–Modulator Reconstruction (TQ-CMR)",
        "parameters": {
            "collector": {
                "J": COLLECTOR_J,
                "Gamma": COLLECTOR_GAMMA,
                "temperatures": COLLECTOR_TEMPERATURES,
            },
            "modulator": {
                "J": MODULATOR_J,
                "temperature": MODULATOR_TEMPERATURE,
                "gammas": MODULATOR_GAMMAS,
                "feedback_strength": FEEDBACK_STRENGTH,
                "edge_sensitivity": EDGE_SENSITIVITY,
                "coupling_floor": COUPLING_FLOOR,
                "transverse_floor": TRANSVERSE_FLOOR,
            },
        },
        "mean_psnr_db": {
            "one_pass_quantum": float(one_pass.mean()),
            "collector_modulator_classical": float(cm_classical.mean()),
            "collector_modulator_quantum": float(cm_quantum.mean()),
            "gaussian": float(gaussian.mean()),
        },
        "paired_effects": {
            "cm_quantum_minus_cm_classical_mean_db": float((cm_quantum - cm_classical).mean()),
            "cm_quantum_vs_cm_classical_pvalue_two_sided": float(quantum_test.pvalue),
            "cm_quantum_minus_one_pass_quantum_mean_db": float((cm_quantum - one_pass).mean()),
            "cm_quantum_vs_one_pass_pvalue_two_sided": float(modulation_test.pvalue),
        },
        "interpretation_limit": (
            "Measured collector feedback makes this a hybrid quantum-control protocol. Exact diagonalization is "
            "classical simulation. Results do not establish novelty or quantum advantage."
        ),
    }
    write_csv(OUTPUT / "exploratory_runs.csv", rows)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_summary_plot(OUTPUT / "collector_modulator_summary.png", rows, image, example)
    print(json.dumps(report, indent=2))
    print(f"Saved collector–modulator prototype to: {OUTPUT}")


if __name__ == "__main__":
    main()
