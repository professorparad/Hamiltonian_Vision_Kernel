from __future__ import annotations

import numpy as np
import pytest
import torch

import hvk
from hvk.api import MODEL_VARIANTS, run_hvk1d
from hvk.config import HVK2DConfig, HVKRunConfig
from hvk.quantum.quantum_model import QuantumModel
from hvk.quantum.symmetric_model import SymmetricQuantumModel
from hvk.training import training


def test_public_api_exports_model_variants():
    assert hvk.MODEL_VARIANTS == ("standard", "symmetric")
    assert MODEL_VARIANTS == hvk.MODEL_VARIANTS
    assert callable(hvk.run_hvk1d)
    assert callable(hvk.run_hvk2d)
    assert hvk.HVKRunConfig is HVKRunConfig
    assert hvk.HVK2DConfig is HVK2DConfig


def test_run_hvk1d_rejects_unknown_model_variant(tmp_path):
    with pytest.raises(ValueError, match="model_variant"):
        run_hvk1d(
            tmp_path / "image.png",
            model_variant="not-a-model",
            save_outputs=False,
        )


def test_train_selects_standard_model(monkeypatch, tmp_path):
    seen = {}

    class FakeStandard(QuantumModel):
        def __init__(self, feature_dim: int, positional_dim: int):
            torch.nn.Module.__init__(self)
            seen["model"] = "standard"
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features, positions):
            batch = features.shape[0]
            observables = torch.zeros(batch, 27, device=features.device) + self.bias
            energies = torch.zeros(batch, device=features.device) + self.bias
            return observables, energies

    monkeypatch.setattr(training, "QuantumModel", FakeStandard)
    monkeypatch.setattr(training, "build_dataset", _fake_dataset)
    monkeypatch.setattr(training, "mps_reconstruct", lambda patch: patch)

    model, _, outputs = training.train(
        image_path=tmp_path / "image.png",
        image_size=64,
        patch_size=64,
        steps=1,
        device="cpu",
        save_outputs=False,
        track_order_parameters=False,
        model_variant="standard",
    )

    assert seen["model"] == "standard"
    assert isinstance(model, FakeStandard)
    assert outputs["model_variant"] == "standard"


def test_train_selects_symmetric_model(monkeypatch, tmp_path):
    seen = {}

    class FakeSymmetric(SymmetricQuantumModel):
        def __init__(self, feature_dim: int, positional_dim: int):
            torch.nn.Module.__init__(self)
            seen["model"] = "symmetric"
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features, positions):
            batch = features.shape[0]
            observables = torch.zeros(batch, 27, device=features.device) + self.bias
            energies = torch.zeros(batch, device=features.device) + self.bias
            return observables, energies

    monkeypatch.setattr(training, "SymmetricQuantumModel", FakeSymmetric)
    monkeypatch.setattr(training, "build_dataset", _fake_dataset)
    monkeypatch.setattr(training, "mps_reconstruct", lambda patch: patch)

    model, _, outputs = training.train(
        image_path=tmp_path / "image.png",
        image_size=64,
        patch_size=64,
        steps=1,
        device="cpu",
        save_outputs=False,
        track_order_parameters=False,
        model_variant="symmetric",
    )

    assert seen["model"] == "symmetric"
    assert isinstance(model, FakeSymmetric)
    assert outputs["model_variant"] == "symmetric"


def test_training_config_is_packaged():
    config = training.load_config(training.DEFAULT_CONFIG_PATH)
    assert config["model_variant"] == "both"
    assert config["track_order_parameters"] is True


def test_config_rejects_unsupported_system_parameters(tmp_path):
    with pytest.raises(ValueError, match="n_qubits=6"):
        HVKRunConfig(image_path=tmp_path / "image.png", n_qubits=8).validate()

    with pytest.raises(ValueError, match="grayscale"):
        HVKRunConfig(image_path=tmp_path / "image.png", image_mode="rgb").validate()

    with pytest.raises(ValueError, match="sinusoidal"):
        HVKRunConfig(image_path=tmp_path / "image.png", encoding="amplitude").validate()


def _fake_dataset(
    image_path,
    image_size=64,
    patch_size=64,
    positional_dim=8,
    device="cpu",
):
    device = training.resolve_device(device) if isinstance(device, str) else device
    image = np.zeros((image_size, image_size), dtype=np.float32)
    patches = np.zeros((1, patch_size, patch_size), dtype=np.float32)
    raw_positions = np.zeros((1, 2), dtype=np.float32)
    features = torch.zeros(1, 8, dtype=torch.float32, device=device)
    positions = torch.zeros(1, positional_dim, dtype=torch.float32, device=device)
    targets = torch.zeros(1, 1, patch_size, patch_size, dtype=torch.float32, device=device)
    return image, patches, raw_positions, features, positions, targets
