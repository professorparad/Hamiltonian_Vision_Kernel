from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import run_ibm_system_size_shots_sweep as sweep
from run_ibm_hvk_probe import build_hvk_circuit, run_on_ibm

MAX_ATTEMPTS = 6
RETRY_SECONDS = 75


def probe_once(token: str | None) -> bool:
    data = np.load(Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz", allow_pickle=False)
    vector = data["patch_vectors"][0]
    circuit = build_hvk_circuit(vector, "hvk1d", 6)
    try:
        backend, job_id, result = run_on_ibm([circuit], None, 100, token, 6)
    except Exception as exc:  # noqa: BLE001
        print(f"probe failed: {exc}")
        return False
    print(f"probe ok: backend={backend} job_id={job_id}")
    return True


def main() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"=== attempt {attempt}/{MAX_ATTEMPTS} ===", flush=True)
        if probe_once(token):
            print("Probe succeeded. Running full system-size/shots sweep now.", flush=True)
            sys.argv = ["run_ibm_system_size_shots_sweep.py"]
            sweep.main()
            print("SWEEP_COMPLETE", flush=True)
            return
        if attempt < MAX_ATTEMPTS:
            print(f"Retrying in {RETRY_SECONDS}s...", flush=True)
            time.sleep(RETRY_SECONDS)
    print("ALL_ATTEMPTS_FAILED", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
