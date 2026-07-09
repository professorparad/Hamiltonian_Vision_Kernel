from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from src.quantum.circuit import n_bonds, n_qubits


def infer_observable_layout(observable_dim: int) -> tuple[int, int, str]:
    if (observable_dim + 3) % 5 == 0:
        inferred_qubits = (observable_dim + 3) // 5
        return inferred_qubits, inferred_qubits - 1, "full"
    if (observable_dim + 1) % 3 == 0:
        inferred_qubits = (observable_dim + 1) // 3
        return inferred_qubits, inferred_qubits - 1, "zz-only"
    return n_qubits, n_bonds, "full"


def observable_slices(observables: np.ndarray) -> dict[str, np.ndarray]:
    inferred_qubits, inferred_bonds, observable_set = infer_observable_layout(
        observables.shape[1]
    )
    zz_start = 2 * inferred_qubits
    xx_start = zz_start + inferred_bonds
    yy_start = xx_start + inferred_bonds
    empty = np.zeros((observables.shape[0], inferred_bonds), dtype=observables.dtype)
    return {
        "z": observables[:, :inferred_qubits],
        "x": observables[:, inferred_qubits:zz_start],
        "zz": observables[:, zz_start:xx_start],
        "xx": empty if observable_set == "zz-only" else observables[:, xx_start:yy_start],
        "yy": (
            empty
            if observable_set == "zz-only"
            else observables[:, yy_start : yy_start + inferred_bonds]
        ),
    }


def compute_order_parameters(
    observables: np.ndarray,
    energies: np.ndarray,
    previous_mean_order: float | None = None,
) -> dict[str, float]:
    parts = observable_slices(observables)
    order = parts["z"].mean(axis=1)
    transverse_order = parts["x"].mean(axis=1)
    lattice_correlation = (
        parts["zz"].mean(axis=1) + parts["xx"].mean(axis=1) + parts["yy"].mean(axis=1)
    ) / 3.0
    mean_order = float(np.mean(order))
    susceptibility = 0.0
    if previous_mean_order is not None:
        susceptibility = abs(mean_order - previous_mean_order)

    return {
        "mean_energy": float(np.mean(energies)),
        "mean_order_parameter": mean_order,
        "mean_abs_order_parameter": float(np.mean(np.abs(order))),
        "mean_transverse_order_parameter": float(np.mean(transverse_order)),
        "mean_total_lattice_correlation": float(np.mean(lattice_correlation)),
        "order_parameter_susceptibility": float(susceptibility),
    }


def detect_phase_transition(epoch_rows: list[dict]) -> dict[str, float | int | bool]:
    if not epoch_rows:
        return {
            "detected": False,
            "critical_epoch": -1,
            "max_susceptibility": 0.0,
            "order_parameter_jump": 0.0,
            "susceptibility_baseline": 0.0,
            "susceptibility_spread": 0.0,
            "susceptibility_threshold": 0.0,
            "proof": "No epoch order-parameter rows were available.",
        }

    susceptibilities = np.array(
        [float(row["order_parameter_susceptibility"]) for row in epoch_rows],
        dtype=np.float32,
    )
    orders = np.array(
        [float(row["mean_order_parameter"]) for row in epoch_rows], dtype=np.float32
    )
    peak_index = int(np.argmax(susceptibilities))
    baseline = float(np.median(susceptibilities))
    spread = float(np.std(susceptibilities))
    threshold = baseline + 2.0 * spread
    max_susceptibility = float(susceptibilities[peak_index])

    detected = bool(max_susceptibility > threshold and max_susceptibility > 0.0)
    critical_epoch = int(epoch_rows[peak_index]["epoch"])
    order_parameter_jump = float(
        abs(orders[peak_index] - orders[max(peak_index - 1, 0)])
    )

    proof = (
        f"Phase transition at epoch {critical_epoch}: "
        f"susceptibility peak {max_susceptibility:.6f} exceeds "
        f"threshold {threshold:.6f} "
        f"(median {baseline:.6f} + 2*std {spread:.6f}); "
        f"order-parameter jump {order_parameter_jump:.6f}."
        if detected
        else f"No phase transition detected: susceptibility peak "
        f"{max_susceptibility:.6f} does not exceed threshold {threshold:.6f}."
    )

    return {
        "detected": detected,
        "critical_epoch": critical_epoch,
        "max_susceptibility": max_susceptibility,
        "order_parameter_jump": order_parameter_jump,
        "susceptibility_baseline": baseline,
        "susceptibility_spread": spread,
        "susceptibility_threshold": threshold,
        "proof": proof,
    }
