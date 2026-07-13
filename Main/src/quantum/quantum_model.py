import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from src.quantum.circuit import (
    n_layers,
)


class QuantumModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        positional_dim: int,
        use_classical_replacement: bool = False,
        use_parameter_matched_classical: bool = False,
        vqc_mode: str = "standard",
        observable_noise: bool = True,
        qubit_count: int = 6,
        observable_set: str = "full",
    ):
        super().__init__()
        if observable_set not in {"full", "zz-only"}:
            raise ValueError("observable_set must be 'full' or 'zz-only'")
        self.n_qubits = qubit_count
        self.n_bonds = qubit_count - 1
        self.observable_set = observable_set
        self.observable_dim = (
            2 * self.n_qubits + self.n_bonds
            if observable_set == "zz-only"
            else 2 * self.n_qubits + 3 * self.n_bonds
        )
        self.use_classical_replacement = use_classical_replacement
        self.use_parameter_matched_classical = use_parameter_matched_classical
        self.vqc_mode = vqc_mode
        self.observable_noise = observable_noise
        self.feature_projection = nn.Linear(feature_dim, self.n_qubits)
        self.position_projection = nn.Linear(positional_dim, self.n_qubits)
        if use_classical_replacement:
            self.classical_map = nn.Linear(self.n_qubits, self.observable_dim)
        elif use_parameter_matched_classical:
            self.classical_down = nn.Linear(self.n_qubits, 1, bias=False)
            self.classical_up = nn.Linear(1, self.observable_dim, bias=False)
        else:
            self.weights = nn.Parameter(
                torch.rand(n_layers, self.n_qubits, 3) * float(np.pi)
            )
            self.Jx = nn.Parameter(0.1 * torch.randn(self.n_bonds))
            self.Jy = nn.Parameter(0.1 * torch.randn(self.n_bonds))
            self.Jz = nn.Parameter(0.1 * torch.randn(self.n_bonds))
        self._vqc = self._build_vqc(entangling=True)
        self._vqc_no_entanglement = self._build_vqc(entangling=False)

    def _build_vqc(self, entangling: bool):
        qdevice = qml.device("default.qubit", wires=self.n_qubits)
        n_qubits = self.n_qubits
        n_bonds = self.n_bonds
        observable_set = self.observable_set

        @qml.qnode(qdevice, interface="torch")
        def circuit(
            inputs: torch.Tensor,
            positional_angles: torch.Tensor,
            weights: torch.Tensor,
        ):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for qubit in range(n_qubits):
                qml.RY(positional_angles[qubit], wires=qubit)
            if entangling:
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            else:
                for layer in range(n_layers):
                    for qubit in range(n_qubits):
                        qml.Rot(
                            weights[layer, qubit, 0],
                            weights[layer, qubit, 1],
                            weights[layer, qubit, 2],
                            wires=qubit,
                        )
            z_obs = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            x_obs = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            zz = [
                qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1))
                for i in range(n_bonds)
            ]
            if observable_set == "zz-only":
                return z_obs + x_obs + zz
            xx = [
                qml.expval(qml.PauliX(i) @ qml.PauliX(i + 1))
                for i in range(n_bonds)
            ]
            yy = [
                qml.expval(qml.PauliY(i) @ qml.PauliY(i + 1))
                for i in range(n_bonds)
            ]
            return z_obs + x_obs + zz + xx + yy

        return circuit

    def forward(self, features: torch.Tensor, positional_encoding: torch.Tensor):
        projected_features = self.feature_projection(features)
        projected_positions = self.position_projection(positional_encoding)
        observables = []
        energies = []

        for feature_vector, position_vector in zip(
            projected_features, projected_positions
        ):
            if self.use_classical_replacement:
                combined = feature_vector + position_vector
                output = torch.tanh(self.classical_map(combined))
                energy = output.new_tensor(0.0)
            elif self.use_parameter_matched_classical:
                combined = feature_vector + position_vector
                output = torch.tanh(self.classical_up(self.classical_down(combined)))
                energy = output.new_tensor(0.0)
            elif self.vqc_mode == "random":
                output = torch.randn(self.observable_dim, device=features.device)
                energy = output.new_tensor(0.0)
            else:
                circuit = (
                    self._vqc_no_entanglement
                    if self.vqc_mode == "no-entanglement"
                    else self._vqc
                )
                output = torch.stack(
                    circuit(feature_vector, position_vector, self.weights)
                )
                zz_start = 2 * self.n_qubits
                xx_start = zz_start + self.n_bonds
                yy_start = xx_start + self.n_bonds

                ZZ = output[zz_start:xx_start]
                if self.observable_set == "zz-only":
                    energy = torch.sum(self.Jz * ZZ)
                else:
                    XX = output[xx_start:yy_start]
                    YY = output[yy_start : yy_start + self.n_bonds]
                    energy = (
                        torch.sum(self.Jz * ZZ)
                        + torch.sum(self.Jx * XX)
                        + torch.sum(self.Jy * YY)
                    )
            observables.append(output)
            energies.append(energy)

        observables = torch.stack(observables).float()
        energies = torch.stack(energies).float()
        if self.training and self.observable_noise:
            observables = observables + 0.01 * torch.randn_like(observables)
        return observables, energies
