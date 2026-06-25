from __future__ import annotations

import numpy as np

from hvk.hvk2d.model import ALL_EDGES, N_QUBITS


def hvk2d_order_summary(observables, energies, previous_order=None):
    z_obs = observables[:, :N_QUBITS]
    x_obs = observables[:, N_QUBITS : 2 * N_QUBITS]
    zz_corr = observables[:, 2 * N_QUBITS :]
    order = z_obs.mean(axis=1)
    mean_order = float(order.mean())
    susceptibility = 0.0 if previous_order is None else abs(mean_order - previous_order)
    return {
        "mean_energy": float(np.mean(energies)),
        "mean_order_parameter": mean_order,
        "order_parameter_susceptibility": float(susceptibility),
        "mean_abs_order_parameter": float(np.mean(np.abs(order))),
        "mean_transverse_order_parameter": float(np.mean(x_obs.mean(axis=1))),
        "mean_total_lattice_correlation": float(np.mean(zz_corr.mean(axis=1))),
    }


def hvk2d_correlation_rows(
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

