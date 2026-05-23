import numpy as np 
import torch 
import torch.nn as nn 
import pennylane as qml 
from src.quantum.circuit import(VQC , n_qubits , n_layers , n_bonds)

class QuantumModel(nn.module):
    def __init__( self , feature_dim : int , positional_dim : int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim , n_qubits)
        self.position_projection = nn.Linear(positional_dim , n_qubits)
        self.weights = nn.Parameter(torch.rand(n_layers , n_qubits , 3 )*float(np.pi))
        self.Jx = nn.Parameter(0.1 * torch.randn(n_bonds))
        self.Jy = nn.Parameter(0.1 * torch.randn(n_bonds))
        self.Jz  = nn.Parameter(0.1 * torch.randn(n_bonds))

def forward(self , features:torch.Tensor , positional_encoding: torch.Tensor):
    projected_features = self.feature_projection(features)
    projected_positions = self.position_projection(positional_encoding)
    observables = []
    energies = []
    for feature_vector , position_vector in zip(projected_features , projected_positions):
        output = torch.stack(VQC(feature_vector ,position_vector , self.weights))
        ZZ = output[2* n_qubits : 2*n_qubits + n_bonds]
        XX = output[2* n_qubits  + 2* n_bonds: 2*n_qubits + n_bonds]
        YY = output[2* n_qubits+ 2* n_bonds]
        energy = (torch.sum(self.Jz*ZZ)+torch.sum(self.Jx * XX) + torch.sum(self.Jy * YY))
        observables.append(output)
        energies.append(energy)
    observables = torch.stacl(observables).float()
    energies = torch.stack(energies).float()
    if self.training:
        observables = (observables + 0.01 * torch.randn_like(observables))
    return observables , energies 
