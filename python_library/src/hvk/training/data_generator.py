from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hvk.quantum.circuit import n_bonds
from hvk.training.order_parameters import (
    compute_order_parameters,
    observable_slices,
)

EPOCH_COLUMNS = [
    "image",
    "epoch",
    "step",
    "reconstructed_image",
    "total_loss",
    "reconstruction_mse",
    "mean_energy",
    "mean_order_parameter",
    "mean_abs_order_parameter",
    "mean_transverse_order_parameter",
    "mean_total_lattice_correlation",
    "order_parameter_susceptibility",
]


CORRELATION_COLUMNS = [
    "image",
    "epoch",
    "step",
    "patch_index",
    "patch_row",
    "patch_col",
    "total_loss",
    "reconstruction_mse",
    "energy",
    "order_parameter",
    "abs_order_parameter",
    "transverse_order_parameter",
    "total_lattice_correlation",
    *[f"corr_ZZ_{i}" for i in range(n_bonds)],
    *[f"corr_XX_{i}" for i in range(n_bonds)],
    *[f"corr_YY_{i}" for i in range(n_bonds)],
]


class TrainingDataGenerator:
    def __init__(self, output_dir: str | Path, image_name: str):
        self.output_dir = Path(output_dir)
        self.image_name = image_name
        self.epoch_rows: list[dict] = []
        self.correlation_rows: list[dict] = []
        self._previous_mean_order: float | None = None

    def record(
        self,
        *,
        epoch: int,
        step: int,
        frame_path: str,
        total_loss: float,
        reconstruction_mse: float,
        observables: np.ndarray,
        energies: np.ndarray,
        positions: np.ndarray,
    ) -> dict[str, float]:
        summary = compute_order_parameters(
            observables=observables,
            energies=energies,
            previous_mean_order=self._previous_mean_order,
        )
        self._previous_mean_order = summary["mean_order_parameter"]

        epoch_row = {
            "image": self.image_name,
            "epoch": epoch,
            "step": step,
            "reconstructed_image": frame_path,
            "total_loss": total_loss,
            "reconstruction_mse": reconstruction_mse,
            **summary,
        }
        self.epoch_rows.append(epoch_row)
        self.correlation_rows.extend(
            build_correlation_rows(
                image_name=self.image_name,
                epoch=epoch,
                step=step,
                total_loss=total_loss,
                reconstruction_mse=reconstruction_mse,
                observables=observables,
                energies=energies,
                positions=positions,
            )
        )
        return summary

    def write_csvs(self) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        epoch_path = self.output_dir / "hvk_epoch_reconstruction_table.csv"
        correlation_path = self.output_dir / "hvk_epoch_correlation_table.csv"
        write_csv(epoch_path, EPOCH_COLUMNS, self.epoch_rows)
        write_csv(correlation_path, CORRELATION_COLUMNS, self.correlation_rows)
        return epoch_path, correlation_path


def build_correlation_rows(
    *,
    image_name: str,
    epoch: int,
    step: int,
    total_loss: float,
    reconstruction_mse: float,
    observables: np.ndarray,
    energies: np.ndarray,
    positions: np.ndarray,
) -> list[dict]:
    parts = observable_slices(observables)
    rows = []
    for patch_index in range(observables.shape[0]):
        zz = parts["zz"][patch_index]
        xx = parts["xx"][patch_index]
        yy = parts["yy"][patch_index]
        order = float(parts["z"][patch_index].mean())
        transverse = float(parts["x"][patch_index].mean())
        lattice_correlation = float((zz.mean() + xx.mean() + yy.mean()) / 3.0)
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
            "transverse_order_parameter": transverse,
            "total_lattice_correlation": lattice_correlation,
        }
        row.update({f"corr_ZZ_{i}": float(value) for i, value in enumerate(zz)})
        row.update({f"corr_XX_{i}": float(value) for i, value in enumerate(xx)})
        row.update({f"corr_YY_{i}": float(value) for i, value in enumerate(yy)})
        rows.append(row)
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
