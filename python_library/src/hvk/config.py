"""Configuration objects for public HVK runners."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OrderParameterFn = Callable[[np.ndarray, np.ndarray, float | None], dict]


@dataclass(frozen=True)
class HVKRunConfig:
    image_path: str | Path
    output_dir: str | Path | None = "hvk_outputs"
    image_size: int = 256
    patch_size: int = 64
    positional_dim: int = 8
    steps: int = 120
    lr: float = 0.003
    device: str = "auto"
    quantum_device: str = "default.qubit"
    n_qubits: int = 6
    n_layers: int = 2
    image_mode: str = "grayscale"
    encoding: str = "sinusoidal"
    save_outputs: bool = True
    show_plots: bool = False
    track_order_parameters: bool = True
    save_epoch_media: bool = True
    epoch_frame_interval: int = 1
    zero_latent_uses_positions: bool = False

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.quantum_device != "default.qubit":
            raise ValueError("Only quantum_device='default.qubit' is currently supported.")
        if self.n_qubits != 6:
            raise ValueError("The current HVK circuits support n_qubits=6.")
        if self.n_layers != 2:
            raise ValueError("The current HVK circuits support n_layers=2.")
        if self.image_mode != "grayscale":
            raise ValueError("Only image_mode='grayscale' is currently supported.")
        if self.encoding != "sinusoidal":
            raise ValueError("Only encoding='sinusoidal' is currently supported.")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")


@dataclass(frozen=True)
class HVK2DConfig(HVKRunConfig):
    lr: float = 0.004
    steps: int = 200
    save_gif: bool = True
    order_parameter_fn: OrderParameterFn | None = None

