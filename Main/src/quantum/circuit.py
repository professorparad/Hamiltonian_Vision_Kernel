import pennylane as qml
import torch

n_qubits = 6
n_layers = 2
n_bonds = n_qubits - 1
observable_dim = n_qubits + n_qubits + 3 * n_bonds
device = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(device, interface="torch")
def VQC(inputs: torch.Tensor, positional_angles: torch.Tensor, weights: torch.Tensor):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for qubit in range(n_qubits):
        qml.RY(positional_angles[qubit], wires=qubit)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    Z = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    X = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    ZZ = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(n_bonds)]
    XX = [qml.expval(qml.PauliX(i) @ qml.PauliX(i + 1)) for i in range(n_bonds)]
    YY = [qml.expval(qml.PauliY(i) @ qml.PauliY(i + 1)) for i in range(n_bonds)]
    return Z + X + ZZ + XX + YY
