"""Workstream 4 (free simulator legs): noise/shot trade-off sweep for the
hardware robustness study.

Reuses the exact already-trained checkpoints from the real hardware pilot
(no retraining). For HVK1D (Monalisa) and HVK2D (4 CIFAR images), builds the
identical Qiskit measurement circuits used for the real hardware submission,
then evaluates each at:
  - ideal (noiseless) statevector simulation -- exact, no shot noise
  - AerSimulator running the real ibm_fez device noise model (pulled from
    live backend properties, not a synthetic model), at shot counts
    256/512/1024/4096

This produces most of the noise-shot trade-off curve at zero IBM Quantum
quota cost (noise-model simulation, not a real-hardware submission), so the
tiny remaining free-tier quota can be spent only on real-hardware anchor
points that confirm the simulated trend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")
MAIN_DIR = REPO_ROOT / "Main"
for p in (MAIN_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.preprocessing.image_loader import load_image_grayscale
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features

OUT_DIR = REPO_ROOT / "IBM_Cloud" / "outputs" / "hardware_robustness_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHOT_COUNTS = [256, 512, 1024, 4096]
N_REPEATS = 3  # repeated noisy-sim executions per shot count, for a reproducibility interval

# ---------------------------------------------------------------------------
# HVK1D (Monalisa)
# ---------------------------------------------------------------------------
HVK1D_CHECKPOINT_DIR = (
    REPO_ROOT / "Main2" / "newHVK" / "results" / "ablation_study"
    / "legacy_hvk_controls" / "eval_controls" / "shared-baseline-seed-42"
)
HVK1D_IMAGE_PATH = MAIN_DIR / "data" / "monalisa.jpg"
N_QUBITS = 6
N_BONDS = N_QUBITS - 1
N_LAYERS = 2
HVK1D_FEATURE_DIM = 46
HVK1D_POSITIONAL_DIM = 8
HVK1D_PATCH_SIZE = 64


class PatchDecoder1D(nn.Module):
    def __init__(self, observable_dim: int, positional_dim: int, patch_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observable_dim + positional_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, patch_size * patch_size), nn.Sigmoid(),
        )

    def forward(self, observables, positional_encoding):
        return self.network(torch.cat([observables, positional_encoding], dim=-1))


class QuantumModelWeights1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Linear(HVK1D_FEATURE_DIM, N_QUBITS)
        self.position_projection = nn.Linear(HVK1D_POSITIONAL_DIM, N_QUBITS)
        self.weights = nn.Parameter(torch.zeros(N_LAYERS, N_QUBITS, 3))
        self.Jx = nn.Parameter(torch.zeros(N_BONDS))
        self.Jy = nn.Parameter(torch.zeros(N_BONDS))
        self.Jz = nn.Parameter(torch.zeros(N_BONDS))


def state_prep_gates_1d(qc, inputs, positional_angles, weights):
    for q in range(N_QUBITS):
        qc.rx(float(inputs[q]), q)
    for q in range(N_QUBITS):
        qc.ry(float(positional_angles[q]), q)
    for layer in range(weights.shape[0]):
        for q in range(N_QUBITS):
            phi, theta, omega = (float(x) for x in weights[layer, q])
            qc.rz(phi, q); qc.ry(theta, q); qc.rz(omega, q)
        ring_range = (layer % (N_QUBITS - 1)) + 1
        for q in range(N_QUBITS):
            qc.cx(q, (q + ring_range) % N_QUBITS)


def build_measurement_circuits_1d(inputs, positional_angles, weights):
    circuits = {}
    for basis in ("Z", "X", "Y"):
        qc = QuantumCircuit(N_QUBITS, N_QUBITS)
        state_prep_gates_1d(qc, inputs, positional_angles, weights)
        if basis == "X":
            for q in range(N_QUBITS):
                qc.h(q)
        elif basis == "Y":
            for q in range(N_QUBITS):
                qc.sdg(q); qc.h(q)
        qc.measure(range(N_QUBITS), range(N_QUBITS))
        circuits[basis] = qc
    return circuits


def observables_from_counts_1d(counts_by_basis):
    def marginals_and_pairs(counts):
        shots = sum(counts.values())
        singles = np.zeros(N_QUBITS); pairs = np.zeros(N_BONDS)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            z = np.array([1.0 if b == "0" else -1.0 for b in bits[:N_QUBITS]])
            singles += z * count
            pairs += np.array([z[i] * z[i + 1] for i in range(N_BONDS)]) * count
        return singles / shots, pairs / shots

    z_s, zz = marginals_and_pairs(counts_by_basis["Z"])
    x_s, xx = marginals_and_pairs(counts_by_basis["X"])
    _, yy = marginals_and_pairs(counts_by_basis["Y"])
    return np.concatenate([z_s, x_s, zz, xx, yy])


def load_hvk1d():
    model = QuantumModelWeights1D()
    model.load_state_dict(torch.load(HVK1D_CHECKPOINT_DIR / "model.pt", map_location="cpu"))
    model.eval()
    decoder = PatchDecoder1D(observable_dim=2 * N_QUBITS + 3 * N_BONDS, positional_dim=HVK1D_POSITIONAL_DIM, patch_size=HVK1D_PATCH_SIZE)
    decoder.load_state_dict(torch.load(HVK1D_CHECKPOINT_DIR / "decoder.pt", map_location="cpu"))
    decoder.eval()

    image = load_image_grayscale(str(HVK1D_IMAGE_PATH), size=(256, 256))
    patches, positions = extract_patches(image, patch_size=HVK1D_PATCH_SIZE)
    features = np.stack([extract_mps_features(p, bond_dim=4) for p in patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_std = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0) + 1e-8)
    positional_encoding = sinusoidal_positional_encoding(positions, d_model=HVK1D_POSITIONAL_DIM)

    with torch.no_grad():
        proj_features = model.feature_projection(features_std)
        proj_positions = model.position_projection(positional_encoding)
    weights_np = model.weights.detach().numpy()
    return {
        "topology": "HVK1D", "image_name": "monalisa", "image": image, "patches": patches, "positions": positions,
        "proj_features": proj_features.detach().numpy(), "proj_positions": proj_positions.detach().numpy(),
        "weights": weights_np, "decoder": decoder, "positional_encoding": positional_encoding,
        "build_circuits": build_measurement_circuits_1d, "observables_from_counts": observables_from_counts_1d,
        "patch_size": HVK1D_PATCH_SIZE, "image_size": 256,
    }


# ---------------------------------------------------------------------------
# HVK2D (CIFAR checkpoints)
# ---------------------------------------------------------------------------
HVK2D_CHECKPOINT_ROOT = REPO_ROOT / "Baselines" / "cifar10_comparisons" / "hvk2d" / "hardware_checkpoints"
HVK2D_EDGES_H = [(0, 1), (1, 2), (3, 4), (4, 5)]
HVK2D_EDGES_V = [(0, 3), (1, 4), (2, 5)]
HVK2D_ALL_EDGES = HVK2D_EDGES_H + HVK2D_EDGES_V
HVK2D_PATCH_SIZE = 8
HVK2D_POSITIONAL_DIM = 4


class PatchDecoder2D(nn.Module):
    def __init__(self, positional_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Linear(19 + positional_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, patch_size * patch_size), nn.Sigmoid(),
        )

    def forward(self, observables, positions):
        return self.net(torch.cat([observables, positions], dim=-1)).view(-1, 1, self.patch_size, self.patch_size)


def state_prep_gates_2d(qc, inputs, positional_angles, weights):
    for q in range(N_QUBITS):
        qc.rx(float(inputs[q]), q)
    for q in range(N_QUBITS):
        qc.ry(float(positional_angles[q]), q)
    for layer in range(weights.shape[0]):
        for source, target in HVK2D_ALL_EDGES:
            qc.cx(source, target)
        for q in range(N_QUBITS):
            phi, theta, omega = (float(x) for x in weights[layer, q])
            qc.rz(phi, q); qc.ry(theta, q); qc.rz(omega, q)


def build_measurement_circuits_2d(inputs, positional_angles, weights):
    circuits = {}
    for basis in ("Z", "X"):
        qc = QuantumCircuit(N_QUBITS, N_QUBITS)
        state_prep_gates_2d(qc, inputs, positional_angles, weights)
        if basis == "X":
            for q in range(N_QUBITS):
                qc.h(q)
        qc.measure(range(N_QUBITS), range(N_QUBITS))
        circuits[basis] = qc
    return circuits


def observables_from_counts_2d(counts_by_basis):
    def marginals(counts):
        shots = sum(counts.values())
        singles = np.zeros(N_QUBITS)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            z = np.array([1.0 if b == "0" else -1.0 for b in bits[:N_QUBITS]])
            singles += z * count
        return singles / shots

    def pair_correlations(counts):
        shots = sum(counts.values())
        pairs = np.zeros(len(HVK2D_ALL_EDGES))
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            z = np.array([1.0 if b == "0" else -1.0 for b in bits[:N_QUBITS]])
            pairs += np.array([z[u] * z[v] for u, v in HVK2D_ALL_EDGES]) * count
        return pairs / shots

    z_singles = marginals(counts_by_basis["Z"])
    x_singles = marginals(counts_by_basis["X"])
    zz = pair_correlations(counts_by_basis["Z"])
    return np.concatenate([z_singles, x_singles, zz])


def load_hvk2d(stem: str):
    ckpt_dir = HVK2D_CHECKPOINT_ROOT / stem
    patches = np.load(ckpt_dir / "patches.npy")
    positions = np.load(ckpt_dir / "positions.npy")

    from Main2.src.model import Quantum2DGridModel

    feature_dim = extract_mps_features(patches[0] + 1e-4, n_sites=6, bond_dim=4).shape[0]
    model = Quantum2DGridModel(feature_dim=feature_dim, positional_dim=HVK2D_POSITIONAL_DIM)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    decoder = PatchDecoder2D(positional_dim=HVK2D_POSITIONAL_DIM, patch_size=HVK2D_PATCH_SIZE)
    decoder.load_state_dict(torch.load(ckpt_dir / "decoder.pt", map_location="cpu"))
    decoder.eval()

    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=6, bond_dim=4) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_std = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positional_encoding = sinusoidal_positional_encoding(positions, d_model=HVK2D_POSITIONAL_DIM)

    with torch.no_grad():
        proj_features = model.feature_projection(features_std)
        proj_positions = model.position_projection(positional_encoding)
    weights_np = model.weights.detach().numpy()

    image = np.zeros((32, 32), dtype=np.float32)  # ground truth reconstructed from patches below
    grid = int(np.sqrt(len(patches)))
    for idx, (r, c) in enumerate(positions):
        rr, cc = int(round(r * 32)), int(round(c * 32))
        image[rr : rr + HVK2D_PATCH_SIZE, cc : cc + HVK2D_PATCH_SIZE] = patches[idx]

    return {
        "topology": "HVK2D", "image_name": stem, "image": image, "patches": patches, "positions": positions,
        "proj_features": proj_features.detach().numpy(), "proj_positions": proj_positions.detach().numpy(),
        "weights": weights_np, "decoder": decoder, "positional_encoding": positional_encoding,
        "build_circuits": build_measurement_circuits_2d, "observables_from_counts": observables_from_counts_2d,
        "patch_size": HVK2D_PATCH_SIZE, "image_size": 32,
    }


# ---------------------------------------------------------------------------
# Shared simulation / metrics
# ---------------------------------------------------------------------------
def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def decode_reconstruction(ckpt: dict, observables_per_patch: np.ndarray) -> np.ndarray:
    obs_t = torch.tensor(observables_per_patch, dtype=torch.float32)
    pos_t = ckpt["positional_encoding"]
    with torch.no_grad():
        if ckpt["topology"] == "HVK1D":
            pred = ckpt["decoder"](obs_t, pos_t).numpy()
            pred = pred.reshape(-1, ckpt["patch_size"], ckpt["patch_size"])
        else:
            pred = ckpt["decoder"](obs_t, pos_t).numpy()[:, 0]

    image_size, patch_size = ckpt["image_size"], ckpt["patch_size"]
    recon = np.zeros((image_size, image_size), dtype=np.float32)
    for idx, (r, c) in enumerate(ckpt["positions"]):
        rr, cc = int(round(r * image_size)), int(round(c * image_size))
        recon[rr : rr + patch_size, cc : cc + patch_size] = pred[idx]
    mse = float(np.mean((recon - ckpt["image"]) ** 2))
    return recon, mse


def ideal_statevector_run(ckpt: dict) -> dict:
    n_patches = len(ckpt["patches"])
    obs_list = []
    for i in range(n_patches):
        circuits = ckpt["build_circuits"](ckpt["proj_features"][i], ckpt["proj_positions"][i], ckpt["weights"])
        counts_by_basis = {}
        for basis, qc in circuits.items():
            sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
            probs = sv.probabilities_dict()
            counts_by_basis[basis] = {bits: int(round(p * 2_000_000)) for bits, p in probs.items()}
        obs_list.append(ckpt["observables_from_counts"](counts_by_basis))
    _, mse = decode_reconstruction(ckpt, np.array(obs_list))
    return {"mode": "ideal_statevector", "shots": None, "mse": mse, "psnr": psnr_from_mse(mse)}


def noisy_aer_run(ckpt: dict, noise_backend, shots: int, seed: int) -> dict:
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    sim = AerSimulator.from_backend(noise_backend)
    n_patches = len(ckpt["patches"])
    obs_list = []
    for i in range(n_patches):
        circuits = ckpt["build_circuits"](ckpt["proj_features"][i], ckpt["proj_positions"][i], ckpt["weights"])
        counts_by_basis = {}
        for basis, qc in circuits.items():
            tqc = transpile(qc, sim, optimization_level=1)
            result = sim.run(tqc, shots=shots, seed_simulator=seed).result()
            counts_by_basis[basis] = result.get_counts()
        obs_list.append(ckpt["observables_from_counts"](counts_by_basis))
    _, mse = decode_reconstruction(ckpt, np.array(obs_list))
    return {"mode": "noisy_aer_ibm_fez", "shots": shots, "seed": seed, "mse": mse, "psnr": psnr_from_mse(mse)}


def main():
    from qiskit_ibm_runtime.fake_provider import FakeFez

    noise_backend = FakeFez()
    print(f"Noise model source: {noise_backend.name}", flush=True)

    loaders = [("HVK1D", "monalisa", load_hvk1d)]
    stems = sorted(p.name for p in HVK2D_CHECKPOINT_ROOT.iterdir() if p.is_dir())
    for stem in stems:
        loaders.append(("HVK2D", stem, lambda s=stem: load_hvk2d(s)))

    results = []
    for topology, name, loader in loaders:
        print(f"\n--- loading {topology} / {name} ---", flush=True)
        ckpt = loader()
        print(f"\n=== {ckpt['topology']} / {ckpt['image_name']} ({len(ckpt['patches'])} patches) ===", flush=True)
        r_ideal = ideal_statevector_run(ckpt)
        r_ideal.update({"topology": ckpt["topology"], "image_name": ckpt["image_name"]})
        print("  ideal:", json.dumps(r_ideal), flush=True)
        results.append(r_ideal)
        (OUT_DIR / "simulator_sweep.json").write_text(json.dumps(results, indent=2))

        for shots in SHOT_COUNTS:
            for rep in range(N_REPEATS):
                r = noisy_aer_run(ckpt, noise_backend, shots, seed=1000 + rep)
                r.update({"topology": ckpt["topology"], "image_name": ckpt["image_name"]})
                print(f"  noisy shots={shots} rep={rep}:", json.dumps(r), flush=True)
                results.append(r)
                (OUT_DIR / "simulator_sweep.json").write_text(json.dumps(results, indent=2))

    print("\nDone. Saved to", OUT_DIR / "simulator_sweep.json")


if __name__ == "__main__":
    main()
