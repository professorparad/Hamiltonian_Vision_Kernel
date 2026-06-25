"""
U(1)-symmetric HVK1D quantum model.

Energy = J*ZZ + K*(XX + YY) where J and K are learnable per-bond parameters.
This mirrors the HVK2D structure where a single J per bond scales ZZ correlations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hvk.quantum.circuit import VQC, n_bonds, n_layers, n_qubits


class SymmetricQuantumModel(nn.Module):
    """U(1)-symmetric HVK1D with learnable J (ZZ coupling) and K (XX=YY coupling).

    The Hamiltonian is:
        H = sum_bonds J_b * ZZ_b  +  sum_bonds K_b * (XX_b + YY_b)

    This enforces U(1) symmetry (conservation of total Sz) since J acts on ZZ
    while XX and YY share the same coupling K, making the model axially symmetric
    in the XY plane — analogous to the HVK2D's structural symmetry.
    """

    def __init__(self, feature_dim: int, positional_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, n_qubits)
        self.position_projection = nn.Linear(positional_dim, n_qubits)
        self.weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * float(np.pi))

        # U(1)-symmetric couplings: J for ZZ, K for XX+YY
        self.J = nn.Parameter(0.1 * torch.randn(n_bonds))   # ZZ coupling
        self.K = nn.Parameter(0.1 * torch.randn(n_bonds))   # XX = YY coupling

    def forward(self, features: torch.Tensor, positional_encoding: torch.Tensor):
        projected_features = self.feature_projection(features)
        projected_positions = self.position_projection(positional_encoding)
        observables = []
        energies = []

        for feature_vector, position_vector in zip(
            projected_features, projected_positions
        ):
            output = torch.stack(VQC(feature_vector, position_vector, self.weights))
            zz_start = 2 * n_qubits
            xx_start = zz_start + n_bonds
            yy_start = xx_start + n_bonds

            ZZ = output[zz_start:xx_start]
            XX = output[xx_start:yy_start]
            YY = output[yy_start : yy_start + n_bonds]

            # U(1)-symmetric: J*ZZ + K*(XX + YY)
            energy = (
                torch.sum(self.J * ZZ)
                + torch.sum(self.K * (XX + YY))
            )
            observables.append(output)
            energies.append(energy)

        observables = torch.stack(observables).float()
        energies = torch.stack(energies).float()
        if self.training:
            observables = observables + 0.01 * torch.randn_like(observables)
        return observables, energies
