# Quantum Components

This folder contains the PennyLane side of HVK1D.

## Files

- `circuit.py`: defines the 6-qubit variational circuit and observable layout.
- `quantum_model.py`: standard HVK1D model with separate `Jx`, `Jy`, and `Jz`
  coupling parameters.
- `symmetric_model.py`: U(1)-symmetric model with `J * ZZ + K * (XX + YY)`.

## Device Selection

`circuit.py` chooses the PennyLane backend in this order:

1. `lightning.gpu`
2. `lightning.qubit`
3. `default.qubit`

This is separate from PyTorch device selection. PyTorch can see CUDA even when
PennyLane cannot run a CUDA simulator.

## Observable Layout

The circuit returns:

- single-qubit Z expectations
- single-qubit X expectations
- nearest-neighbor ZZ correlations
- nearest-neighbor XX correlations
- nearest-neighbor YY correlations

The models use those observables to compute the Hamiltonian energy term.
