"""Replay a freshly-trained HVK2D CIFAR checkpoint's forward pass on real IBM
Quantum hardware and decode the hardware-measured observables back to pixels.

Uses checkpoints produced by
Baselines/cifar10_comparisons/hvk2d/train_and_save_hvk2d_checkpoints.py.
The HVK2D grid circuit (Main2/src/model.py) only measures Z, X, and ZZ
observables (no XX/YY), so each patch needs only two measurement bases
(Z-basis gives Z singles + ZZ pairs; X-basis gives X singles).
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
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Main"))
sys.path.insert(0, str(REPO_ROOT / "Baselines" / "cifar10_comparisons"))

from src.preprocessing.positional_encoding import sinusoidal_positional_encoding

N_QUBITS = 6
N_LAYERS = 2
EDGES_H = [(0, 1), (1, 2), (3, 4), (4, 5)]
EDGES_V = [(0, 3), (1, 4), (2, 5)]
ALL_EDGES = EDGES_H + EDGES_V
OBS_DIM = N_QUBITS + N_QUBITS + len(ALL_EDGES)
PATCH_SIZE = 8
POSITIONAL_DIM = 4

CHECKPOINT_ROOT = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "hvk2d" / "hardware_checkpoints"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hvk2d_cifar_hardware_reconstruction"


class Quantum2DGridWeights(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, N_QUBITS)
        self.position_projection = nn.Linear(POSITIONAL_DIM, N_QUBITS)
        self.weights = nn.Parameter(torch.zeros(N_LAYERS, N_QUBITS, 3))
        self.j_2d = nn.Parameter(torch.zeros(len(ALL_EDGES)))


class PatchDecoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + POSITIONAL_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, PATCH_SIZE * PATCH_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, observables, positions):
        combined = torch.cat([observables, positions], dim=-1)
        return self.net(combined).view(-1, 1, PATCH_SIZE, PATCH_SIZE)


def state_prep_gates(
    qc: QuantumCircuit, inputs: np.ndarray, positional_angles: np.ndarray, weights: np.ndarray
) -> None:
    """Matches quantum_grid_circuit exactly: AngleEmbedding(rotation='X' default) +
    RY(positional) + N_LAYERS x [CNOT over EDGES_H+EDGES_V, then Rot(phi,theta,omega) per qubit]."""
    for q in range(N_QUBITS):
        qc.rx(float(inputs[q]), q)
    for q in range(N_QUBITS):
        qc.ry(float(positional_angles[q]), q)
    for layer in range(weights.shape[0]):
        for source, target in ALL_EDGES:
            qc.cx(source, target)
        for q in range(N_QUBITS):
            phi, theta, omega = (float(x) for x in weights[layer, q])
            qc.rz(phi, q)
            qc.ry(theta, q)
            qc.rz(omega, q)


def build_measurement_circuits(
    inputs: np.ndarray, positional_angles: np.ndarray, weights: np.ndarray
) -> dict[str, QuantumCircuit]:
    circuits = {}
    for basis in ("Z", "X"):
        qc = QuantumCircuit(N_QUBITS, N_QUBITS)
        state_prep_gates(qc, inputs, positional_angles, weights)
        if basis == "X":
            for q in range(N_QUBITS):
                qc.h(q)
        qc.measure(range(N_QUBITS), range(N_QUBITS))
        circuits[basis] = qc
    return circuits


def observables_from_counts(counts_by_basis: dict[str, dict[str, int]]) -> np.ndarray:
    def marginals(counts: dict[str, int]):
        shots = sum(counts.values())
        singles = np.zeros(N_QUBITS)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            z = np.array([1.0 if b == "0" else -1.0 for b in bits[:N_QUBITS]])
            singles += z * count
        return singles / shots, counts

    def pair_correlations(counts: dict[str, int]):
        shots = sum(counts.values())
        pairs = np.zeros(len(ALL_EDGES))
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            z = np.array([1.0 if b == "0" else -1.0 for b in bits[:N_QUBITS]])
            pairs += np.array([z[u] * z[v] for u, v in ALL_EDGES]) * count
        return pairs / shots

    z_singles, z_counts = marginals(counts_by_basis["Z"])
    x_singles, _ = marginals(counts_by_basis["X"])
    zz = pair_correlations(counts_by_basis["Z"])
    return np.concatenate([z_singles, x_singles, zz])


def local_statevector_observables(inputs: np.ndarray, positional_angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    circuits = build_measurement_circuits(inputs, positional_angles, weights)
    result = {}
    for basis, qc in circuits.items():
        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        probs = sv.probabilities_dict()
        result[basis] = {bits: int(round(p * 1_000_000)) for bits, p in probs.items()}
    return observables_from_counts(result)


def run_image(stem: str, args, backend=None) -> dict:
    ckpt_dir = CHECKPOINT_ROOT / stem
    patches = np.load(ckpt_dir / "patches.npy")
    positions = np.load(ckpt_dir / "positions.npy")
    cached_observables = np.load(ckpt_dir / "observables.npy")
    n_patches = min(args.max_patches, len(patches))

    model = Quantum2DGridWeights(feature_dim=46)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    model.feature_projection = nn.Linear(state["feature_projection.weight"].shape[1], N_QUBITS)
    model.load_state_dict(state)
    model.eval()

    decoder = PatchDecoder2D()
    decoder.load_state_dict(torch.load(ckpt_dir / "decoder.pt", map_location="cpu"))
    decoder.eval()

    from src.tensornetworks.mps_features import extract_mps_features

    features = np.array([extract_mps_features(p, n_sites=6, bond_dim=4) for p in patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_std = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positional_encoding = sinusoidal_positional_encoding(positions, d_model=POSITIONAL_DIM)

    with torch.no_grad():
        projected_features = model.feature_projection(features_std)
        projected_positions = model.position_projection(positional_encoding)

    weights_np = model.weights.detach().numpy()
    qk_observables = []
    qk_err = 0.0
    for i in range(n_patches):
        out = local_statevector_observables(
            projected_features[i].detach().numpy(), projected_positions[i].detach().numpy(), weights_np
        )
        qk_observables.append(out)
        qk_err = max(qk_err, float(np.max(np.abs(out - cached_observables[i]))))
    qk_observables = np.array(qk_observables)
    print(f"[{stem}] Qiskit vs cached observables.npy, max abs error over {n_patches} patches: {qk_err:.2e}")

    n_circuits = 2 * n_patches
    print(f"[{stem}] plan: {n_patches} patches x 2 bases (Z/X) = {n_circuits} circuits, {args.shots} shots each.")

    target = torch.tensor(patches[:n_patches], dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        recon_sim = decoder(torch.tensor(qk_observables, dtype=torch.float32), positional_encoding[:n_patches])
    mse_sim = float(torch.mean((recon_sim - target) ** 2))
    psnr_sim = 20 * np.log10(1.0 / np.sqrt(mse_sim)) if mse_sim > 0 else float("inf")

    result = {
        "stem": stem,
        "n_patches": n_patches,
        "psnr_simulator_db": psnr_sim,
        "mse_simulator": mse_sim,
    }

    if not args.submit:
        return result

    all_circuits = []
    circuit_index = []
    for i in range(n_patches):
        circuits = build_measurement_circuits(
            projected_features[i].detach().numpy(), projected_positions[i].detach().numpy(), weights_np
        )
        for basis, qc in circuits.items():
            all_circuits.append(qc)
            circuit_index.append((i, basis))

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    transpiled = transpile(all_circuits, backend=backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run(transpiled, shots=args.shots)
    print(f"[{stem}] job_id={job.job_id()}")
    job_result = job.result()

    counts_by_patch: dict[int, dict[str, dict[str, int]]] = {i: {} for i in range(n_patches)}
    for idx, (patch_i, basis) in enumerate(circuit_index):
        pub_result = job_result[idx]
        data = getattr(pub_result, "data", pub_result)
        creg = data.c if hasattr(data, "c") else data.meas
        counts_by_patch[patch_i][basis] = creg.get_counts()

    hw_observables = np.array([observables_from_counts(counts_by_patch[i]) for i in range(n_patches)])
    with torch.no_grad():
        recon_hw = decoder(torch.tensor(hw_observables, dtype=torch.float32), positional_encoding[:n_patches])
    mse_hw = float(torch.mean((recon_hw - target) ** 2))
    psnr_hw = 20 * np.log10(1.0 / np.sqrt(mse_hw)) if mse_hw > 0 else float("inf")

    result.update(
        {"backend": backend.name, "job_id": job.job_id(), "psnr_hardware_db": psnr_hw, "mse_hardware": mse_hw}
    )
    np.save(OUTPUT_DIR / f"{stem}_hw_observables.npy", hw_observables)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-patches", type=int, default=16)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--backend")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--allow-large-job", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name for p in CHECKPOINT_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(stems)} trained CIFAR checkpoints: {stems}")

    backend = None
    if args.submit:
        n_circuits = 2 * args.max_patches * len(stems)
        if n_circuits > 20 and not args.allow_large_job:
            raise ValueError(f"This would submit {n_circuits} circuits total. Pass --allow-large-job to proceed.")
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        if args.backend:
            backend = service.backend(args.backend)
        else:
            candidates = service.backends(operational=True, simulator=False, min_num_qubits=N_QUBITS)
            backend = sorted(candidates, key=lambda b: getattr(b.status(), "pending_jobs", 999999))[0]
        print(f"Using backend: {backend.name}")

    all_results = [run_image(stem, args, backend) for stem in stems]
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
