"""Quantum transverse-field Ising image encoding and completion experiment.

This is an exact statevector/density-matrix simulation of a quantum thermal
model. It deliberately has no neural network, MPS feature extractor, or
classical image decoder. Classical code is still necessarily used to load the
image, simulate the quantum system, aggregate measurements, and save figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_IMAGE = REPO_ROOT / "Main" / "data" / "monalisa.jpg"
DEFAULT_OUTPUT = HERE / "results"

I2 = np.eye(2, dtype=np.float64)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def operator_on_site(op: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    result = np.array([[1.0]], dtype=np.float64)
    for q in range(n_qubits):
        result = np.kron(result, op if q == site else I2)
    return result


def build_operators(
    patch_height: int, patch_width: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    n_qubits = patch_height * patch_width
    z_ops = [operator_on_site(Z, i, n_qubits) for i in range(n_qubits)]
    x_ops = [operator_on_site(X, i, n_qubits) for i in range(n_qubits)]
    bonds: list[tuple[int, int]] = []
    for row in range(patch_height):
        for col in range(patch_width):
            i = row * patch_width + col
            if col + 1 < patch_width:
                bonds.append((i, i + 1))
            if row + 1 < patch_height:
                bonds.append((i, i + patch_width))
    return z_ops, x_ops, bonds


def thermal_observables(
    fields: np.ndarray,
    temperatures: list[float],
    coupling: float,
    transverse_field: float,
    z_ops: list[np.ndarray],
    x_ops: list[np.ndarray],
    bonds: list[tuple[int, int]],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    dimension = z_ops[0].shape[0]
    hamiltonian = np.zeros((dimension, dimension), dtype=np.float64)
    zz_ops: dict[tuple[int, int], np.ndarray] = {}

    for i, j in bonds:
        zz = z_ops[i] @ z_ops[j]
        zz_ops[(i, j)] = zz
        hamiltonian -= coupling * zz
    for i, field in enumerate(fields):
        hamiltonian -= field * z_ops[i]
        hamiltonian -= transverse_field * x_ops[i]

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    def expected(op: np.ndarray, probabilities: np.ndarray) -> float:
        diagonal_in_energy_basis = np.sum(eigenvectors * (op @ eigenvectors), axis=0)
        return float(probabilities @ diagonal_in_energy_basis)

    outputs: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for temperature in temperatures:
        shifted = eigenvalues - eigenvalues.min()
        weights = np.exp(-shifted / temperature)
        probabilities = weights / weights.sum()

        z_expectations = np.array([expected(op, probabilities) for op in z_ops])
        x_expectations = np.array([expected(op, probabilities) for op in x_ops])
        local_energy = -fields * z_expectations - transverse_field * x_expectations
        for i, j in bonds:
            bond_energy = -coupling * expected(zz_ops[(i, j)], probabilities)
            local_energy[i] += 0.5 * bond_energy
            local_energy[j] += 0.5 * bond_energy
        outputs[temperature] = (z_expectations, local_energy)
    return outputs


def load_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size[1], size[0]), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255.0


def make_completion_mask(shape: tuple[int, int], observed_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < observed_fraction
    # Keep a thin boundary as context, a standard image-completion condition.
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def reconstruct(
    image: np.ndarray,
    observed_mask: np.ndarray,
    temperatures: list[float],
    patch_shape: tuple[int, int],
    coupling: float,
    transverse_field: float,
    field_strength: float,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    patch_h, patch_w = patch_shape
    z_ops, x_ops, bonds = build_operators(patch_h, patch_w)
    reconstructions = {temperature: np.zeros_like(image) for temperature in temperatures}
    heat_maps = {temperature: np.zeros_like(image) for temperature in temperatures}

    for top in range(0, image.shape[0], patch_h):
        for left in range(0, image.shape[1], patch_w):
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
            for temperature, (z_expectations, local_energy) in measured.items():
                reconstructions[temperature][top : top + patch_h, left : left + patch_w] = (
                    (1.0 + z_expectations) / 2.0
                ).reshape(patch_shape)
                heat_maps[temperature][top : top + patch_h, left : left + patch_w] = local_energy.reshape(
                    patch_shape
                )
    return reconstructions, heat_maps


def metrics(target: np.ndarray, prediction: np.ndarray, evaluation_mask: np.ndarray) -> dict[str, float]:
    difference = target[evaluation_mask] - prediction[evaluation_mask]
    mse = float(np.mean(difference**2))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(difference))),
        "psnr_db": float(10.0 * math.log10(1.0 / max(mse, 1e-15))),
    }


def save_gray(path: Path, array: np.ndarray) -> None:
    pixels = np.uint8(np.clip(array, 0.0, 1.0) * 255.0 + 0.5)
    Image.fromarray(pixels, mode="L").save(path)


def save_summary(
    output_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    reconstructions: dict[str, dict[float, np.ndarray]],
    heat_maps: dict[str, dict[float, np.ndarray]],
    temperatures: list[float],
) -> None:
    columns = 2 + len(temperatures)
    fig, axes = plt.subplots(5, columns, figsize=(3.0 * columns, 13.0), constrained_layout=True)
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Mona Lisa target")
    axes[0, 1].imshow(np.where(mask, image, 0.5), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Completion input")
    for col in range(2, columns):
        axes[0, col].axis("off")

    for task_row, task in enumerate(("encoded", "completion")):
        row = 1 + 2 * task_row
        axes[row, 0].text(0.5, 0.5, f"{task}\nmagnetization", ha="center", va="center")
        axes[row + 1, 0].text(0.5, 0.5, f"{task}\nlocal energy", ha="center", va="center")
        axes[row, 0].axis("off")
        axes[row + 1, 0].axis("off")
        axes[row, 1].axis("off")
        axes[row + 1, 1].axis("off")
        all_heat = np.concatenate([heat_maps[task][temperature].ravel() for temperature in temperatures])
        heat_limit = max(abs(float(all_heat.min())), abs(float(all_heat.max())))
        for index, temperature in enumerate(temperatures, start=2):
            axes[row, index].imshow(reconstructions[task][temperature], cmap="gray", vmin=0, vmax=1)
            axes[row, index].set_title(f"T={temperature:g}")
            axes[row + 1, index].imshow(
                heat_maps[task][temperature],
                cmap="coolwarm",
                vmin=-heat_limit,
                vmax=heat_limit,
            )
            axes[row + 1, index].set_title(f"energy map, T={temperature:g}")

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Quantum transverse-field Ising thermal image experiment", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--patch-height", type=int, default=2)
    parser.add_argument("--patch-width", type=int, default=3)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--coupling", type=float, default=0.45)
    parser.add_argument("--transverse-field", type=float, default=0.35)
    parser.add_argument("--field-strength", type=float, default=2.0)
    parser.add_argument("--observed-fraction", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % args.patch_height or args.width % args.patch_width:
        raise ValueError("Image dimensions must be divisible by patch dimensions.")
    if any(temperature <= 0 for temperature in args.temperatures):
        raise ValueError("All temperatures must be positive.")

    args.output.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image, (args.height, args.width))
    completion_mask = make_completion_mask(image.shape, args.observed_fraction, args.seed)
    task_masks = {
        "encoded": np.ones_like(completion_mask, dtype=bool),
        "completion": completion_mask,
    }

    all_reconstructions: dict[str, dict[float, np.ndarray]] = {}
    all_heat_maps: dict[str, dict[float, np.ndarray]] = {}
    report: dict[str, object] = {
        "interpretation": (
            "Exact classical simulation of a quantum Gibbs model; reconstruction is from site-resolved "
            "magnetizations, while temperature controls thermal mixing. It is not inversion from temperature alone."
        ),
        "image": str(args.image),
        "image_shape": list(image.shape),
        "patch_shape": [args.patch_height, args.patch_width],
        "qubits_per_patch": args.patch_height * args.patch_width,
        "parameters": {
            "temperatures": args.temperatures,
            "coupling_J": args.coupling,
            "transverse_field_Gamma": args.transverse_field,
            "local_field_strength": args.field_strength,
            "observed_fraction_requested": args.observed_fraction,
            "observed_fraction_actual": float(completion_mask.mean()),
            "seed": args.seed,
        },
        "tasks": {},
    }

    save_gray(args.output / "target.png", image)
    save_gray(args.output / "completion_observed_input.png", np.where(completion_mask, image, 0.5))

    for task, observed_mask in task_masks.items():
        reconstructions, heat_maps = reconstruct(
            image,
            observed_mask,
            args.temperatures,
            (args.patch_height, args.patch_width),
            args.coupling,
            args.transverse_field,
            args.field_strength,
        )
        all_reconstructions[task] = reconstructions
        all_heat_maps[task] = heat_maps
        evaluation_mask = np.ones_like(observed_mask, dtype=bool) if task == "encoded" else ~observed_mask
        task_results: dict[str, object] = {}
        for temperature in args.temperatures:
            label = f"{temperature:g}".replace(".", "p")
            save_gray(args.output / f"{task}_reconstruction_T{label}.png", reconstructions[temperature])
            np.save(args.output / f"{task}_local_energy_T{label}.npy", heat_maps[temperature])
            task_results[str(temperature)] = metrics(image, reconstructions[temperature], evaluation_mask)
        report["tasks"][task] = task_results

    save_summary(
        args.output / "quantum_spin_temperature_summary.png",
        image,
        completion_mask,
        all_reconstructions,
        all_heat_maps,
        args.temperatures,
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved private experiment results to: {args.output}")


if __name__ == "__main__":
    main()
