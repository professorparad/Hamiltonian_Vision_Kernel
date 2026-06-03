import torch
import torch.nn as nn


class PatchDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int | None = None,
        patch_size: int = 64,
        observable_dim: int | None = None,
        positional_dim: int | None = None,
    ):
        super().__init__()
        if input_dim is None:
            if observable_dim is None or positional_dim is None:
                raise ValueError(
                    "Provide input_dim or both observable_dim and positional_dim"
                )
            input_dim = observable_dim + positional_dim

        self.patch_size = patch_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size * patch_size),
            nn.Sigmoid()
        )

    def forward(self, observables: torch.Tensor, positional_encoding: torch.Tensor):
        x = torch.cat([observables, positional_encoding], dim=1)
        x = self.network(x)
        return x.view(-1, 1, self.patch_size, self.patch_size)
