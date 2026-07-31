import torch
import torch.nn as nn


class PatchDecoder(nn.Module):
    """Patch decoder with an optional Hamiltonian-energy input.

    In the original decoder, the learned energy never reaches reconstruction at
    all -- it only exists as a side-channel loss term, so it can never help
    reconstruction, only compete with it (see Main_new/README.md). Setting
    `use_energy_feature=True` concatenates the (bounded, see quantum_model.py)
    per-patch energy into the decoder input, making it an actual feature the
    decoder can learn to use, not just a regularizer sitting outside the
    reconstruction pathway.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        patch_size: int = 64,
        observable_dim: int | None = None,
        positional_dim: int | None = None,
        use_energy_feature: bool = False,
    ):
        super().__init__()
        self.use_energy_feature = use_energy_feature
        if input_dim is None:
            if observable_dim is None or positional_dim is None:
                raise ValueError(
                    "Provide input_dim or both observable_dim and positional_dim"
                )
            input_dim = observable_dim + positional_dim
        if use_energy_feature:
            input_dim += 1

        self.patch_size = patch_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size * patch_size),
            nn.Sigmoid(),
        )

    def forward(
        self,
        observables: torch.Tensor,
        positional_encoding: torch.Tensor,
        energy: torch.Tensor | None = None,
    ):
        parts = [observables, positional_encoding]
        if self.use_energy_feature:
            if energy is None:
                raise ValueError("use_energy_feature=True requires an `energy` tensor")
            parts.append(energy.reshape(-1, 1))
        x = torch.cat(parts, dim=1)
        x = self.network(x)
        return x.view(-1, 1, self.patch_size, self.patch_size)
