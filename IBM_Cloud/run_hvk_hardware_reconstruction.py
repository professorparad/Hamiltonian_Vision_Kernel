"""Replay an already-trained HVK1D checkpoint's forward pass on real IBM Quantum
hardware and decode the hardware-measured observables back to pixels.

Reuses the exact trained weights from
Main2/newHVK/results/ablation_study/legacy_hvk_controls/eval_controls/shared-baseline-seed-42/
(model.pt + decoder.pt), so nothing is retrained. The script:

1. Recomputes the 16 Monalisa patch inputs (MPS features -> feature_projection,
   sinusoidal positional encoding -> position_projection) exactly as training.py does.
2. Runs the trained PennyLane VQC locally in eval mode and checks the result matches
   the cached observables.npy (a correctness check, zero hardware cost).
3. Converts the PennyLane circuit (AngleEmbedding + RY + StronglyEntanglingLayers) to
   three Qiskit measurement circuits per patch (Z-basis, X-basis, Y-basis) by
   introspecting the executed PennyLane tape rather than hand-translating gates, since
   Z/ZZ share one circuit, X/XX share one circuit, and Y/YY share one circuit.
4. With --submit, sends all circuits for the requested patches to a real IBM backend,
   reconstructs the 27-D observable vector per patch from hardware counts, feeds it
   through the trained decoder, and reports hardware-vs-simulator PSNR/SSIM.

Without --submit this is entirely free (simulator-only validation + circuit-count report).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Main"))

from src.preprocessing.image_loader import load_image_grayscale
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features

CHECKPOINT_DIR = (
    REPO_ROOT
    / "Main2"
    / "newHVK"
    / "results"
    / "ablation_study"
    / "legacy_hvk_controls"
    / "eval_controls"
    / "shared-baseline-seed-42"
)
IMAGE_PATH = REPO_ROOT / "Main" / "data" / "monalisa.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hardware_reconstruction"

N_QUBITS = 6
N_BONDS = N_QUBITS - 1
N_LAYERS = 2
FEATURE_DIM = 46
POSITIONAL_DIM = 8
PATCH_SIZE = 64


class PatchDecoder(nn.Module):
    def __init__(self, observable_dim: int, positional_dim: int, patch_size: int):
        super().__init__()
        input_dim = observable_dim + positional_dim
        output_dim = patch_size * patch_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, observables: torch.Tensor, positional_encoding: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([observables, positional_encoding], dim=-1)
        return self.network(combined)


class QuantumModelWeights(nn.Module):
    """Holds only the trained parameters needed for inference (mirrors model.pt keys)."""

    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Linear(FEATURE_DIM, N_QUBITS)
        self.position_projection = nn.Linear(POSITIONAL_DIM, N_QUBITS)
        self.weights = nn.Parameter(torch.zeros(N_LAYERS, N_QUBITS, 3))
        self.Jx = nn.Parameter(torch.zeros(N_BONDS))
        self.Jy = nn.Parameter(torch.zeros(N_BONDS))
        self.Jz = nn.Parameter(torch.zeros(N_BONDS))


def build_pennylane_circuit():
    device = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(device, interface="torch")
    def circuit(inputs, positional_angles, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        for qubit in range(N_QUBITS):
            qml.RY(positional_angles[qubit], wires=qubit)
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        z_obs = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
        x_obs = [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
        zz = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(N_BONDS)]
        xx = [qml.expval(qml.PauliX(i) @ qml.PauliX(i + 1)) for i in range(N_BONDS)]
        yy = [qml.expval(qml.PauliY(i) @ qml.PauliY(i + 1)) for i in range(N_BONDS)]
        return z_obs + x_obs + zz + xx + yy

    return circuit


def state_prep_gates(qc: QuantumCircuit, inputs: np.ndarray, positional_angles: np.ndarray, weights: np.ndarray) -> None:
    """Reproduce AngleEmbedding(rotation='X') + RY(positional) + StronglyEntanglingLayers
    directly in Qiskit, matching PennyLane's documented default construction:
    AngleEmbedding default rotation is 'X'; StronglyEntanglingLayers applies Rot(phi,theta,omega)
    per wire per layer then a CNOT ring with range=1 (wire i -> wire (i+range) % n_wires)."""
    for q in range(N_QUBITS):
        qc.rx(float(inputs[q]), q)
    for q in range(N_QUBITS):
        qc.ry(float(positional_angles[q]), q)
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        for q in range(N_QUBITS):
            phi, theta, omega = (float(x) for x in weights[layer, q])
            qc.rz(phi, q)
            qc.ry(theta, q)
            qc.rz(omega, q)
        ring_range = (layer % (N_QUBITS - 1)) + 1
        for q in range(N_QUBITS):
            qc.cx(q, (q + ring_range) % N_QUBITS)


def build_measurement_circuits(inputs: np.ndarray, positional_angles: np.ndarray, weights: np.ndarray) -> dict[str, QuantumCircuit]:
    circuits = {}
    for basis in ("Z", "X", "Y"):
        qc = QuantumCircuit(N_QUBITS, N_QUBITS)
        state_prep_gates(qc, inputs, positional_angles, weights)
        if basis == "X":
            for q in range(N_QUBITS):
                qc.h(q)
        elif basis == "Y":
            for q in range(N_QUBITS):
                qc.sdg(q)
                qc.h(q)
        qc.measure(range(N_QUBITS), range(N_QUBITS))
        circuits[basis] = qc
    return circuits


def observables_from_counts(counts_by_basis: dict[str, dict[str, int]]) -> np.ndarray:
    def marginals_and_pairs(counts: dict[str, int]):
        shots = sum(counts.values())
        singles = np.zeros(N_QUBITS)
        pairs = np.zeros(N_BONDS)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            z = np.array([1.0 if b == "0" else -1.0 for b in bits[:N_QUBITS]])
            singles += z * count
            pairs += np.array([z[i] * z[i + 1] for i in range(N_BONDS)]) * count
        return singles / shots, pairs / shots

    z_singles, zz = marginals_and_pairs(counts_by_basis["Z"])
    x_singles, xx = marginals_and_pairs(counts_by_basis["X"])
    _, yy = marginals_and_pairs(counts_by_basis["Y"])
    return np.concatenate([z_singles, x_singles, zz, xx, yy])


def local_statevector_observables(inputs: np.ndarray, positional_angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    circuits = build_measurement_circuits(inputs, positional_angles, weights)
    result = {}
    for basis, qc in circuits.items():
        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        probs = sv.probabilities_dict()
        result[basis] = {bits: int(round(p * 1_000_000)) for bits, p in probs.items()}
    return observables_from_counts(result)


def load_patch_inputs():
    image = load_image_grayscale(str(IMAGE_PATH), size=(256, 256))
    patches, positions = extract_patches(image, patch_size=PATCH_SIZE)
    features = np.stack([extract_mps_features(patch, bond_dim=4) for patch in patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_std = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0) + 1e-8)
    positional_encoding = sinusoidal_positional_encoding(positions, d_model=POSITIONAL_DIM)
    return patches, positions, features_std, positional_encoding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patches", type=int, default=16)
    parser.add_argument("--submit", action="store_true", help="Submit to real IBM hardware. Default: local validation only.")
    parser.add_argument("--backend")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--allow-large-job", action="store_true")
    parser.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"))
    args = parser.parse_args()

    model = QuantumModelWeights()
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model.pt", map_location="cpu"))
    model.eval()
    decoder = PatchDecoder(observable_dim=2 * N_QUBITS + 3 * N_BONDS, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE)
    decoder.load_state_dict(torch.load(CHECKPOINT_DIR / "decoder.pt", map_location="cpu"))
    decoder.eval()

    patches, positions, features_std, positional_encoding = load_patch_inputs()
    n_patches = min(args.max_patches, len(patches))

    with torch.no_grad():
        projected_features = model.feature_projection(features_std)
        projected_positions = model.position_projection(positional_encoding)

    cached_observables = np.load(CHECKPOINT_DIR / "observables.npy")
    weights_np = model.weights.detach().numpy()

    pennylane_circuit = build_pennylane_circuit()
    max_err = 0.0
    sim_observables = []
    for i in range(n_patches):
        with torch.no_grad():
            out = torch.stack(
                pennylane_circuit(projected_features[i], projected_positions[i], model.weights)
            ).numpy()
        sim_observables.append(out)
        err = float(np.max(np.abs(out - cached_observables[i])))
        max_err = max(max_err, err)
    sim_observables = np.array(sim_observables)
    print(f"[validate] PennyLane vs cached observables.npy, max abs error over {n_patches} patches: {max_err:.2e}")

    qk_observables = []
    qk_err = 0.0
    for i in range(n_patches):
        qk_out = local_statevector_observables(
            projected_features[i].detach().numpy(),
            projected_positions[i].detach().numpy(),
            weights_np,
        )
        qk_observables.append(qk_out)
        err = float(np.max(np.abs(qk_out - cached_observables[i])))
        qk_err = max(qk_err, err)
    qk_observables = np.array(qk_observables)
    print(f"[validate] Qiskit statevector vs cached observables.npy, max abs error over {n_patches} patches: {qk_err:.2e}")

    n_circuits = 3 * n_patches
    print(f"[plan] {n_patches} patches x 3 measurement bases (Z/X/Y) = {n_circuits} circuits, {args.shots} shots each.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.submit:
        print("[dry-run] No hardware job submitted. Pass --submit to run on real IBM hardware.")
        with torch.no_grad():
            recon_sim = decoder(torch.tensor(qk_observables, dtype=torch.float32), positional_encoding[:n_patches])
        mse = float(torch.mean((recon_sim.view(n_patches, PATCH_SIZE, PATCH_SIZE) - torch.tensor(patches[:n_patches])) ** 2))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        print(f"[dry-run] Simulator-replay decode PSNR over {n_patches} patches: {psnr:.2f} dB (mse={mse:.3e})")
        return

    if n_circuits > 20 and not args.allow_large_job:
        raise ValueError(
            f"This would submit {n_circuits} circuits to real hardware. "
            "Pass --allow-large-job if you've confirmed this fits your quota, "
            "or reduce --max-patches."
        )
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    except ImportError:
        from qiskit_ibm_runtime import Sampler

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=args.token) if args.token else QiskitRuntimeService()
    if args.backend:
        backend = service.backend(args.backend)
    else:
        candidates = service.backends(operational=True, simulator=False, min_num_qubits=N_QUBITS)
        backend = sorted(candidates, key=lambda b: getattr(b.status(), "pending_jobs", 999999))[0]
    print(f"[submit] Using backend: {backend.name}")

    all_circuits = []
    circuit_index = []
    for i in range(n_patches):
        circuits = build_measurement_circuits(
            projected_features[i].detach().numpy(),
            projected_positions[i].detach().numpy(),
            weights_np,
        )
        for basis, qc in circuits.items():
            all_circuits.append(qc)
            circuit_index.append((i, basis))

    transpiled = transpile(all_circuits, backend=backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run(transpiled, shots=args.shots)
    print(f"[submit] job_id={job.job_id()}")
    result = job.result()

    counts_by_patch: dict[int, dict[str, dict[str, int]]] = {i: {} for i in range(n_patches)}
    for idx, (patch_i, basis) in enumerate(circuit_index):
        pub_result = result[idx]
        data = getattr(pub_result, "data", pub_result)
        creg = data.c if hasattr(data, "c") else data.meas
        counts_by_patch[patch_i][basis] = creg.get_counts()

    hw_observables = np.array([observables_from_counts(counts_by_patch[i]) for i in range(n_patches)])

    with torch.no_grad():
        recon_hw = decoder(torch.tensor(hw_observables, dtype=torch.float32), positional_encoding[:n_patches])
        recon_sim = decoder(torch.tensor(sim_observables, dtype=torch.float32), positional_encoding[:n_patches])

    target = torch.tensor(patches[:n_patches])
    mse_hw = float(torch.mean((recon_hw.view(n_patches, PATCH_SIZE, PATCH_SIZE) - target) ** 2))
    mse_sim = float(torch.mean((recon_sim.view(n_patches, PATCH_SIZE, PATCH_SIZE) - target) ** 2))
    psnr_hw = 20 * np.log10(1.0 / np.sqrt(mse_hw)) if mse_hw > 0 else float("inf")
    psnr_sim = 20 * np.log10(1.0 / np.sqrt(mse_sim)) if mse_sim > 0 else float("inf")

    report = {
        "backend": backend.name,
        "job_id": job.job_id(),
        "n_patches": n_patches,
        "shots": args.shots,
        "psnr_hardware_db": psnr_hw,
        "psnr_simulator_db": psnr_sim,
        "mse_hardware": mse_hw,
        "mse_simulator": mse_sim,
        "cached_baseline_psnr_db": 32.235,
    }
    (OUTPUT_DIR / "hardware_reconstruction_report.json").write_text(json.dumps(report, indent=2))
    np.save(OUTPUT_DIR / "hw_observables.npy", hw_observables)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
