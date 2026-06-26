# IBM Cloud HVK Probe

This folder contains small IBM Quantum experiments inspired by the HVK models.
It is intentionally not the full image-reconstruction training loop. Full HVK
training is too large and too iterative for a free-plan QPU workflow.

Instead, these scripts build compact HVK-style circuits from tiny image patches,
submit or dry-run those circuits, and compute simple order parameters from
measurement counts.

## What Is In This Folder

- `prepare_ibm_dataset.py`: creates a small `.npz` patch dataset from Mona Lisa
  or already-downloaded CIFAR images.
- `run_ibm_hvk_probe.py`: builds HVK1D/HVK2D-style circuits and optionally sends
  them to IBM Quantum.
- `requirements-ibm.txt`: Qiskit Runtime dependencies.
- `datasets/`: tiny prepared datasets.
- `outputs/`: generated circuit summaries and hardware-run results.

## What The Probe Measures

The script computes lightweight hardware-friendly signals:

- mean Z order
- mean absolute Z order
- nearest-neighbor or lattice ZZ correlations
- a proxy loss based on order and correlation

This is not neural training loss. It is a small measurement-side proxy that can
fit into limited QPU time.

## Install IBM Runtime Support

```powershell
python -m pip install -r IBM_Cloud\requirements-ibm.txt
```

## Account Setup

Save an IBM Quantum token once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_IBM_QUANTUM_TOKEN",
    set_as_default=True,
)
```

Do not commit tokens. If a token is pasted into a file, log, or chat, rotate it
from IBM before using it for real jobs.

## Prepare Data

Mona Lisa:

```powershell
python IBM_Cloud\prepare_ibm_dataset.py --source monalisa --max-patches 1
```

CIFAR, after running the CIFAR downloader:

```powershell
python IBM_Cloud\prepare_ibm_dataset.py --source cifar --max-patches 1
```

## Dry Run

Use this before submitting anything to hardware:

```powershell
python IBM_Cloud\run_ibm_hvk_probe.py --dataset IBM_Cloud\datasets\monalisa_patches.npz --variant both --dry-run
```

## Submit A Small Job

Start tiny:

```powershell
python IBM_Cloud\run_ibm_hvk_probe.py --dataset IBM_Cloud\datasets\monalisa_patches.npz --variant both --n-qubits 6 --shots 100 --max-patches 1
```

Optionally choose a backend:

```powershell
python IBM_Cloud\run_ibm_hvk_probe.py --backend ibm_brisbane --n-qubits 6 --shots 100 --max-patches 1
```

## Practical Notes

- Keep `--max-patches 1` for the first real run.
- Keep `--shots 100` until you know your account quota and queue behavior.
- Larger qubit counts increase transpilation time, noise, and queue risk.
- This folder is for hardware probing, not for reproducing the full benchmark.
