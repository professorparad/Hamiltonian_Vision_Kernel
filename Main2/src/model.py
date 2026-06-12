from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


N_QUBITS = 6
N_LAYERS = 2
EDGES_H = [(0, 1), (1, 2), (3, 4), (4, 5)]
EDGES_V = [(0, 3), (1, 4), (2, 5)]
ALL_EDGES = EDGES_H + EDGES_V
OBS_DIM = N_QUBITS + N_QUBITS + len(ALL_EDGES)
DEVICE = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(DEVICE, interface="torch")
def quantum_grid_circuit(inputs, positional_angles, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    for qubit in range(N_QUBITS):
        qml.RY(positional_angles[qubit], wires=qubit)
    for layer in range(weights.shape[0]):
        for source, target in EDGES_H:
            qml.CNOT(wires=[source, target])
        for source, target in EDGES_V:
            qml.CNOT(wires=[source, target])
        for qubit in range(N_QUBITS):
            qml.Rot(
                weights[layer, qubit, 0],
                weights[layer, qubit, 1],
                weights[layer, qubit, 2],
                wires=qubit,
            )
    return (
        [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
        + [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
        + [qml.expval(qml.PauliZ(u) @ qml.PauliZ(v)) for u, v in ALL_EDGES]
    )


class Quantum2DGridModel(nn.Module):
    def __init__(self, feature_dim: int, positional_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, N_QUBITS)
        self.position_projection = nn.Linear(positional_dim, N_QUBITS)
        self.weights = nn.Parameter(torch.rand(N_LAYERS, N_QUBITS, 3) * float(np.pi))
        self.j_2d = nn.Parameter(0.1 * torch.randn(len(ALL_EDGES)))

    def forward(self, features, positions):
        vectors = self.feature_projection(features)
        position_angles = self.position_projection(positions)
        observables = []
        energies = []
        for vector, angles in zip(vectors, position_angles):
            output = torch.stack(quantum_grid_circuit(vector, angles, self.weights))
            zz_2d = output[2 * N_QUBITS :]
            observables.append(output)
            energies.append(torch.sum(self.j_2d * zz_2d))
        observables = torch.stack(observables).float()
        energies = torch.stack(energies).float()
        if self.training:
            observables = observables + 0.01 * torch.randn_like(observables)
        return observables, energies


class PatchDecoder(nn.Module):
    def __init__(self, positional_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + positional_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size * patch_size),
            nn.Sigmoid(),
        )

    def forward(self, observables, positions):
        decoder_input = torch.cat([observables, positions], dim=-1)
        return self.net(decoder_input).view(-1, 1, self.patch_size, self.patch_size)
