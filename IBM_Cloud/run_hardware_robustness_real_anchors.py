"""Workstream 4 (real-hardware anchors, quota-budgeted).

Confirms the simulator-based noise/shot sweep (run_hardware_robustness_simulator_sweep.py)
against real IBM Quantum hardware at a small number of anchor points, reusing
the same already-trained checkpoints as the original five-image pilot (no
retraining). Budget-conscious by design: two checkpoints (Monalisa/HVK1D,
CIFAR cat/HVK2D), two shot counts (256 -- matches the original pilot exactly
-- and 1024, a new higher-shot anchor) on the primary backend, plus one
cross-backend confirmation job.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile

REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")
MAIN_DIR = REPO_ROOT / "Main"
for p in (MAIN_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.preprocessing.image_loader import load_image_grayscale
from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features

sys.path.insert(0, str(REPO_ROOT / "IBM_Cloud"))
from run_hardware_robustness_simulator_sweep import (
    load_hvk1d, load_hvk2d, decode_reconstruction, psnr_from_mse,
)

OUT_DIR = REPO_ROOT / "IBM_Cloud" / "outputs" / "hardware_robustness_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_on_hardware(ckpt: dict, backend, shots: int) -> dict:
    from qiskit_ibm_runtime import SamplerV2

    n_patches = len(ckpt["patches"])
    all_circuits, circuit_index = [], []
    for i in range(n_patches):
        circuits = ckpt["build_circuits"](ckpt["proj_features"][i], ckpt["proj_positions"][i], ckpt["weights"])
        for basis, qc in circuits.items():
            all_circuits.append(qc)
            circuit_index.append((i, basis))

    transpiled = transpile(all_circuits, backend=backend, optimization_level=1)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=shots)
    job_id = job.job_id()
    print(f"  submitted job_id={job_id} ({len(all_circuits)} circuits, {shots} shots)", flush=True)
    result = job.result()

    counts_by_patch: dict[int, dict[str, dict[str, int]]] = {i: {} for i in range(n_patches)}
    for idx, (patch_i, basis) in enumerate(circuit_index):
        pub_result = result[idx]
        data = getattr(pub_result, "data", pub_result)
        creg = data.c if hasattr(data, "c") else data.meas
        counts_by_patch[patch_i][basis] = creg.get_counts()

    obs_list = [ckpt["observables_from_counts"](counts_by_patch[i]) for i in range(n_patches)]
    _, mse = decode_reconstruction(ckpt, np.array(obs_list))
    return {
        "mode": "real_hardware", "backend": backend.name, "shots": shots, "job_id": job_id,
        "n_circuits": len(all_circuits), "mse": mse, "psnr": psnr_from_mse(mse),
        "topology": ckpt["topology"], "image_name": ckpt["image_name"],
    }


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    usage_before = service.usage()
    print("Usage before:", usage_before["usage_remaining_seconds"], "s remaining", flush=True)

    kingston = service.backend("ibm_kingston")

    monalisa = load_hvk1d()
    cat = load_hvk2d("0000_cat_domestic_cat_s_000907")

    # kingston has had the shortest queue throughout (3-6 pending vs.
    # 9-51 on marrakesh/fez); the one already-completed marrakesh point
    # (Monalisa, 256 shots, kept below) stays as the cross-backend anchor.
    jobs = [
        (monalisa, kingston, 1024),
        (cat, kingston, 256),
        (cat, kingston, 1024),
    ]

    result_file = OUT_DIR / "real_hardware_anchors.json"
    results = json.loads(result_file.read_text()) if result_file.exists() else []
    done = {(r["topology"], r["image_name"], r["backend"], r["shots"]) for r in results}

    for ckpt, backend, shots in jobs:
        key = (ckpt["topology"], ckpt["image_name"], backend.name, shots)
        if key in done:
            print(f"skip (already done): {key}", flush=True)
            continue
        print(f"\n=== {ckpt['topology']} / {ckpt['image_name']} on {backend.name} @ {shots} shots ===", flush=True)
        r = run_on_hardware(ckpt, backend, shots)
        print("  result:", json.dumps(r), flush=True)
        results.append(r)
        result_file.write_text(json.dumps(results, indent=2))

    usage_after = service.usage()
    print("\nUsage after:", usage_after["usage_remaining_seconds"], "s remaining", flush=True)
    print("Quota consumed this run:", usage_before["usage_remaining_seconds"] - usage_after["usage_remaining_seconds"], "s", flush=True)
    print("\nDone. Saved to", OUT_DIR / "real_hardware_anchors.json")


if __name__ == "__main__":
    main()
