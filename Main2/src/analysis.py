from __future__ import annotations

import numpy as np

from Main2.src.model import ALL_EDGES, N_QUBITS


def main2_order_summary(observables, energies, previous_order=None):
    z_obs = observables[:, :N_QUBITS]
    x_obs = observables[:, N_QUBITS : 2 * N_QUBITS]
    zz_corr = observables[:, 2 * N_QUBITS :]
    order = z_obs.mean(axis=1)
    mean_order = float(order.mean())
    susceptibility = 0.0
    if previous_order is not None:
        susceptibility = abs(mean_order - previous_order)
    return {
        "mean_energy": float(np.mean(energies)),
        "mean_order_parameter": mean_order,
        "order_parameter_susceptibility": float(susceptibility),
        "mean_abs_order_parameter": float(np.mean(np.abs(order))),
        "mean_transverse_order_parameter": float(np.mean(x_obs.mean(axis=1))),
        "mean_total_lattice_correlation": float(np.mean(zz_corr.mean(axis=1))),
    }


def main2_correlation_rows(
    *,
    image_name,
    epoch,
    step,
    total_loss,
    reconstruction_mse,
    observables,
    energies,
    positions,
):
    z_obs = observables[:, :N_QUBITS]
    x_obs = observables[:, N_QUBITS : 2 * N_QUBITS]
    zz_corr = observables[:, 2 * N_QUBITS :]
    rows = []
    for patch_index in range(observables.shape[0]):
        order = float(z_obs[patch_index].mean())
        row = {
            "image": image_name,
            "epoch": epoch,
            "step": step,
            "patch_index": patch_index,
            "patch_row": int(positions[patch_index, 0]),
            "patch_col": int(positions[patch_index, 1]),
            "total_loss": total_loss,
            "reconstruction_mse": reconstruction_mse,
            "energy": float(energies[patch_index]),
            "order_parameter": order,
            "abs_order_parameter": abs(order),
            "transverse_order_parameter": float(x_obs[patch_index].mean()),
            "total_lattice_correlation": float(zz_corr[patch_index].mean()),
        }
        row.update(
            {
                f"corr_ZZ_{source}_{target}": float(value)
                for (source, target), value in zip(ALL_EDGES, zz_corr[patch_index])
            }
        )
        rows.append(row)
    return rows


def detect_phase_transition(epoch_rows):
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
    susceptibility = np.array(
        [row["order_parameter_susceptibility"] for row in epoch_rows],
        dtype=np.float32,
    )
    order = np.array(
        [row["mean_order_parameter"] for row in epoch_rows],
        dtype=np.float32,
    )
    peak = int(np.argmax(susceptibility))
    baseline = float(np.median(susceptibility))
    spread = float(np.std(susceptibility))
    threshold = baseline + 2.0 * spread
    max_susceptibility = float(susceptibility[peak])
    critical_epoch = int(epoch_rows[peak]["epoch"])
    order_parameter_jump = float(abs(order[peak] - order[max(peak - 1, 0)]))
    detected = bool(max_susceptibility > threshold and max_susceptibility > 0.0)
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
