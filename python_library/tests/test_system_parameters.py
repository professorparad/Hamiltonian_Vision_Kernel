from __future__ import annotations

import numpy as np

from hvk.preprocessing.patching import extract_patches
from hvk.preprocessing.positional_encoding import sinusoidal_positional_encoding
from hvk.hvk2d.order_parameters import hvk2d_order_summary
from hvk.training.order_parameters import compute_order_parameters, detect_phase_transition


def test_patch_size_controls_number_of_positions():
    image = np.zeros((64, 64), dtype=np.float32)
    patches, positions = extract_patches(image, patch_size=32)

    assert patches.shape == (4, 32, 32)
    assert positions.shape == (4, 2)


def test_positional_dim_controls_encoding_width():
    positions = np.array([[0, 0], [1, 1]], dtype=np.float32)
    encoded = sinusoidal_positional_encoding(positions, d_model=8)

    assert tuple(encoded.shape) == (2, 8)


def test_order_parameter_summary_uses_previous_order_for_susceptibility():
    observables = np.zeros((2, 27), dtype=np.float32)
    observables[:, :6] = 0.5
    energies = np.array([1.0, 3.0], dtype=np.float32)

    summary = compute_order_parameters(observables, energies, previous_mean_order=0.25)

    assert summary["mean_order_parameter"] == 0.5
    assert summary["mean_energy"] == 2.0
    assert summary["order_parameter_susceptibility"] == 0.25


def test_phase_transition_handles_empty_rows():
    transition = detect_phase_transition([])

    assert transition["detected"] is False
    assert transition["critical_epoch"] == -1


def test_hvk2d_order_parameter_includes_lattice_correlation():
    observables = np.zeros((2, 19), dtype=np.float32)
    observables[:, :6] = 0.25
    observables[:, 6:12] = 0.5
    observables[:, 12:] = 0.75
    energies = np.array([1.0, 2.0], dtype=np.float32)

    summary = hvk2d_order_summary(observables, energies, previous_order=0.1)

    assert summary["mean_order_parameter"] == 0.25
    assert summary["mean_transverse_order_parameter"] == 0.5
    assert summary["mean_total_lattice_correlation"] == 0.75
    assert summary["order_parameter_susceptibility"] == 0.15


def test_hvk2d_custom_order_parameter_hook():
    def custom_order(observables, energies, previous_order):
        return {
            "mean_energy": float(energies.mean()),
            "mean_order_parameter": 42.0,
            "order_parameter_susceptibility": 0.0,
        }

    observables = np.zeros((1, 19), dtype=np.float32)
    energies = np.array([3.0], dtype=np.float32)

    assert custom_order(observables, energies, None)["mean_order_parameter"] == 42.0
