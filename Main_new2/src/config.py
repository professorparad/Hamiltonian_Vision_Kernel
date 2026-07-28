from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAIN2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MAIN2_ROOT.parent


@dataclass
class Main2Config:
    image_path: Path = REPO_ROOT / "Main_new" / "data" / "monalisa.jpg"
    output_dir: Path = MAIN2_ROOT / "outputs" / "training_analysis"
    image_size: int = 256
    patch_size: int = 64
    positional_dim: int = 8
    steps: int = 200
    lr: float = 0.004
    device: str = "auto"
    frame_interval: int = 1
    save_gif: bool = True
    energy_loss_mode: str = "positive"
    energy_weight: float = 0.01
    energy_margin: float = 0.25
