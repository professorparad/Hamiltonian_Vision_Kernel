"""TQ-CMR v3: finite-reservoir, heat-current-driven image reconstruction.

Uses the thermodynamic-neuron dynamics
    d beta / dt = -(beta**2 / C) * (J_collector + J_modulator)
with qubit occupations g(beta)=1/(1+exp(beta*epsilon)).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_experiment import DEFAULT_IMAGE, DEFAULT_OUTPUT, load_image, make_completion_mask, metrics
from run_overlapping_patches import reconstruct_overlapping
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_rel

OUTPUT = DEFAULT_OUTPUT / "heat_current_tqcmr_v3"
IMAGE_SHAPE = (24, 24)
PATCH_SHAPE = (2, 3)
SEEDS = [3, 7, 11, 19, 29, 41]
OBSERVED_FRACTION = 0.55

# Frozen quantum collector from the successful held-out experiment.
COLLECTOR_J = 0.7
COLLECTOR_T = 1.25
COLLECTOR_GAMMA_QUANTUM = 0.35
COLLECTOR_GAMMA_CLASSICAL = 0.0
FIELD_STRENGTH = 2.0

# Thermodynamic-neuron current model.
EPSILON = 1.0
MU_COLLECTOR = 1.0
MU_MODULATOR = 0.3
HEAT_CAPACITY = 10.0
T_FINAL = 4000.0

# The modulator is a local thermal reservoir constructed only from observed
# pixels. Normalized convolution prevents missing pixels acting as zero-valued
# cold reservoirs.
MODULATOR_SIGMA = 1.0


def occupation(beta: np.ndarray) -> np.ndarray:
    values = np.clip(beta * EPSILON, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(values))


def inverse_occupation(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log((1.0 - probability) / probability) / EPSILON


def spatial_modulator_reservoir(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    numerator = gaussian_filter(image * mask, sigma=MODULATOR_SIGMA, mode="nearest")
    denominator = gaussian_filter(mask.astype(np.float64), sigma=MODULATOR_SIGMA, mode="nearest")
    return numerator / np.maximum(denominator, 1e-12)


def evolve_finite_reservoirs(
    collector_target: np.ndarray,
    modulator_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """Integrate all pixel inverse temperatures in one uncoupled vector ODE."""
    shape = collector_target.shape
    collector_flat = np.clip(collector_target.reshape(-1), 1e-6, 1.0 - 1e-6)
    modulator_flat = np.clip(modulator_target.reshape(-1), 1e-6, 1.0 - 1e-6)
    beta_initial = inverse_occupation(collector_flat)

    def derivative(_time: float, beta: np.ndarray) -> np.ndarray:
        output_population = occupation(beta)
        # Current is positive when heat flows INTO the finite output reservoir.
        # This orientation makes the zero-current solution dynamically stable
        # under d_beta/dt=-(beta^2/C)J.
        collector_current = MU_COLLECTOR * EPSILON * (collector_flat - output_population)
        modulator_current = MU_MODULATOR * EPSILON * (modulator_flat - output_population)
        return -(beta**2 / HEAT_CAPACITY) * (collector_current + modulator_current)

    sample_times = np.linspace(0.0, T_FINAL, 101)
    solution = solve_ivp(
        derivative,
        (0.0, T_FINAL),
        beta_initial,
        t_eval=sample_times,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    beta_final = solution.y[:, -1]
    reconstructed = occupation(beta_final).reshape(shape)
    collector_current = MU_COLLECTOR * EPSILON * (collector_flat - reconstructed.reshape(-1))
    modulator_current = MU_MODULATOR * EPSILON * (modulator_flat - reconstructed.reshape(-1))
    net_current = collector_current + modulator_current
    diagnostics = {
        "max_abs_net_current": float(np.max(np.abs(net_current))),
        "mean_abs_net_current": float(np.mean(np.abs(net_current))),
        "solver_steps_recorded": len(solution.t),
        "solver_function_evaluations": int(solution.nfev),
    }
    # Mean occupation trajectory is a compact thermodynamic convergence trace.
    mean_trajectory = occupation(solution.y).mean(axis=0)
    return reconstructed, diagnostics, mean_trajectory


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    path: Path,
    image: np.ndarray,
    example: dict[str, np.ndarray],
    rows: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.5), constrained_layout=True)
    methods = ["collector_quantum", "v3_classical", "v3_quantum", "spatial_modulator"]
    labels = ["Quantum\ncollector", "v3 Γ=0\ncollector", "v3 Γ=.35\ncollector", "Spatial\nreservoir"]
    means = [np.mean([row[name] for row in rows]) for name in methods]
    errors = [np.std([row[name] for row in rows], ddof=1) for name in methods]
    axes[0, 0].bar(labels, means, yerr=errors, capsize=4, color=["#5479a8", "#5b9a72", "#8a4fa3", "#777777"])
    axes[0, 0].set_ylabel("Missing-pixel PSNR (dB)")
    axes[0, 0].set_title("Exploratory masks")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Target")
    axes[0, 2].imshow(example["collector"], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Quantum collector")
    axes[0, 3].imshow(example["modulator"], cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title("Modulator reservoir")
    axes[1, 0].imshow(example["observed"], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Observed input")
    axes[1, 1].imshow(example["v3_classical"], cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("v3 classical collector")
    axes[1, 2].imshow(example["v3_quantum"], cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("v3 quantum collector")
    axes[1, 3].plot(np.linspace(0, T_FINAL, len(example["trajectory"])), example["trajectory"], color="#8a4fa3")
    axes[1, 3].set_xlabel("Thermodynamic time")
    axes[1, 3].set_ylabel("Mean output occupation")
    axes[1, 3].set_title("Finite-reservoir evolution")
    axes[1, 3].grid(alpha=0.25)
    for axis in axes.flat[:7]:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("TQ-CMR v3: heat-current-driven qubit reconstruction", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = load_image(DEFAULT_IMAGE, IMAGE_SHAPE)
    rows: list[dict] = []
    example: dict[str, np.ndarray] = {}
    convergence: list[dict] = []

    for index, seed in enumerate(SEEDS, start=1):
        print(f"TQ-CMR v3 mask {index}/{len(SEEDS)} seed={seed}", flush=True)
        mask = make_completion_mask(image.shape, OBSERVED_FRACTION, seed)
        missing = ~mask
        modulator = spatial_modulator_reservoir(image, mask)
        predictions: dict[str, np.ndarray] = {}
        trajectories: dict[str, np.ndarray] = {}

        for name, gamma in (
            ("v3_classical", COLLECTOR_GAMMA_CLASSICAL),
            ("v3_quantum", COLLECTOR_GAMMA_QUANTUM),
        ):
            collector, _ = reconstruct_overlapping(
                image,
                mask,
                [COLLECTOR_T],
                PATCH_SHAPE,
                COLLECTOR_J,
                gamma,
                FIELD_STRENGTH,
            )
            collector_image = collector[COLLECTOR_T]
            prediction, diagnostics, trajectory = evolve_finite_reservoirs(collector_image, modulator)
            predictions[name] = prediction
            trajectories[name] = trajectory
            convergence.append({"seed": seed, "model": name, **diagnostics})
            if name == "v3_quantum":
                predictions["collector_quantum"] = collector_image

        row = {"seed": seed}
        for name, prediction in {
            "collector_quantum": predictions["collector_quantum"],
            "v3_classical": predictions["v3_classical"],
            "v3_quantum": predictions["v3_quantum"],
            "spatial_modulator": modulator,
        }.items():
            row[name] = metrics(image, prediction, missing)["psnr_db"]
        rows.append(row)
        if index == 1:
            example = {
                "observed": np.where(mask, image, 0.5),
                "collector": predictions["collector_quantum"],
                "modulator": modulator,
                "v3_classical": predictions["v3_classical"],
                "v3_quantum": predictions["v3_quantum"],
                "trajectory": trajectories["v3_quantum"],
            }

    collector = np.array([row["collector_quantum"] for row in rows])
    classical = np.array([row["v3_classical"] for row in rows])
    quantum = np.array([row["v3_quantum"] for row in rows])
    modulator = np.array([row["spatial_modulator"] for row in rows])
    quantum_control_test = ttest_rel(quantum, classical)
    architecture_test = ttest_rel(quantum, collector)
    report = {
        "status": "exploratory thermodynamic-neuron integration",
        "architecture": "TQ-CMR v3 finite-reservoir collector-modulator",
        "equations": {
            "occupation": "g(beta)=1/(1+exp(beta*epsilon))",
            "collector_current": "Jc=mu*epsilon*(g(beta_collector)-g(beta_z)), current into output reservoir",
            "modulator_current": "Jm=mu_prime*epsilon*(g(beta_modulator)-g(beta_z)), current into output reservoir",
            "reservoir_dynamics": "d beta_z/dt=-(beta_z^2/C)*(Jc+Jm)",
        },
        "parameters": {
            "epsilon": EPSILON,
            "mu": MU_COLLECTOR,
            "mu_prime": MU_MODULATOR,
            "heat_capacity": HEAT_CAPACITY,
            "t_final": T_FINAL,
            "collector_J": COLLECTOR_J,
            "collector_temperature": COLLECTOR_T,
            "quantum_Gamma": COLLECTOR_GAMMA_QUANTUM,
        },
        "mean_psnr_db": {
            "one_pass_quantum_collector": float(collector.mean()),
            "v3_classical_collector": float(classical.mean()),
            "v3_quantum_collector": float(quantum.mean()),
            "spatial_modulator_reservoir": float(modulator.mean()),
        },
        "paired_effects": {
            "v3_quantum_minus_v3_classical_db": float((quantum - classical).mean()),
            "quantum_control_pvalue_two_sided": float(quantum_control_test.pvalue),
            "v3_quantum_minus_one_pass_collector_db": float((quantum - collector).mean()),
            "architecture_pvalue_two_sided": float(architecture_test.pvalue),
            "v3_quantum_minus_spatial_modulator_db": float((quantum - modulator).mean()),
        },
        "convergence": {
            "maximum_final_abs_net_current": max(row["max_abs_net_current"] for row in convergence),
            "mean_final_abs_net_current": float(np.mean([row["mean_abs_net_current"] for row in convergence])),
        },
        "interpretation_limit": (
            "The quantum collector is exactly simulated and the spatial reservoir is constructed classically from "
            "observed pixels. This is a hybrid thermodynamic reconstruction protocol, not quantum advantage. The "
            "reference repository's current sign was reversed so its beta ODE has a stable zero-current fixed point."
        ),
    }
    save_csv(OUTPUT / "exploratory_runs.csv", rows)
    save_csv(OUTPUT / "convergence.csv", convergence)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plot(OUTPUT / "tqcmr_v3_summary.png", image, example, rows)
    print(json.dumps(report, indent=2))
    print(f"Saved TQ-CMR v3 results to: {OUTPUT}")


if __name__ == "__main__":
    main()
