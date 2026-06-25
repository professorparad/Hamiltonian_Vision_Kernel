# IBM Cloud HVK Probe

Small IBM Quantum experiments for HVK1D and HVK2D.

This is intentionally separate from the full training code. A full image-reconstruction
training loop is too large for a free-plan QPU workflow, so this folder runs compact
HVK-style circuits on tiny image patches and records order parameters from hardware
measurement counts.

## What This Runs

- `hvk1d`: chain entanglement with configurable qubit count.
- `hvk2d`: grid entanglement with configurable qubit count.
- Dataset choices:
  - `monalisa`: resized Mona Lisa patches.
  - `cifar`: a few already-downloaded CIFAR-10 32x32 grayscale samples.

Outputs are written to `IBM_Cloud/outputs/`.

## Install IBM Runtime Support

```bash
source .venv/bin/activate
pip install -r IBM_Cloud/requirements-ibm.txt
```

## IBM Account Setup

Create an IBM Quantum account/instance and save your token once:

```bash
python - <<'PY'
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_IBM_QUANTUM_TOKEN",
    set_as_default=True,
)
PY
```

You can also avoid saving credentials by setting an environment variable:

```bash
export IBM_QUANTUM_TOKEN="YOUR_IBM_QUANTUM_TOKEN"
```

Do not commit API keys or account email addresses to this repository. If a token was
pasted into chat, logs, or a file, rotate it from IBM Cloud before using it for real jobs.

## Prepare a Tiny Dataset

Mona Lisa:

```bash
python IBM_Cloud/prepare_ibm_dataset.py --source monalisa --max-patches 1
```

CIFAR, after running the CIFAR downloader:

```bash
python IBM_Cloud/prepare_ibm_dataset.py --source cifar --max-patches 1
```

## Dry Run Locally

Build circuits and write metadata without submitting jobs:

```bash
python IBM_Cloud/run_ibm_hvk_probe.py --dataset IBM_Cloud/datasets/monalisa_patches.npz --variant both --dry-run
```

## Submit to IBM Quantum

The free plan can be very tight. Start with one Mona Lisa patch, both variants, and
100 shots. That builds two circuits total:

```bash
python IBM_Cloud/run_ibm_hvk_probe.py \
  --dataset IBM_Cloud/datasets/monalisa_patches.npz \
  --variant both \
  --n-qubits 6 \
  --shots 100 \
  --max-patches 1
```

Optionally pick a backend:

```bash
python IBM_Cloud/run_ibm_hvk_probe.py --backend ibm_brisbane --n-qubits 6 --shots 100 --max-patches 1
```

## Larger IBM Devices

IBM backends may expose many more qubits than the conservative 6-qubit default.
You can request more qubits:

```bash
python IBM_Cloud/run_ibm_hvk_probe.py \
  --dataset IBM_Cloud/datasets/monalisa_patches.npz \
  --variant hvk2d \
  --n-qubits 27 \
  --shots 100 \
  --max-patches 1 \
  --allow-large-free-plan-job
```

For the free plan, start with 6 qubits first. Larger qubit counts increase
transpile time, queue risk, circuit width, and hardware noise. The script maps
image-patch values across however many qubits you request.

## Visual Outputs

The runner writes:

- `circuits_summary.json`
- `circuit_summary.png`
- `ibm_hvk_probe_results.json` after a hardware run
- `ibm_hvk_probe_metrics.png` after a hardware run

`ibm_hvk_probe_metrics.png` includes measured order parameters and a lightweight
hardware-proxy loss:

```text
proxy_loss = (1 - mean_abs_order_parameter) + 0.5 * (1 - mean_zz_correlation)
```

This is not full neural training loss. It is a hardware-friendly signal that can
be produced within the free-plan runtime window.

## Notes

- Keep `--max-patches` at `1` for the first free-plan run.
- Keep `--shots` at `100` for the first free-plan run.
- By default, the runner refuses jobs larger than 2 circuits or 100 shots. Use
  `--allow-large-free-plan-job` only after confirming your account quota and queue time.
- The scripts compute order parameters from measured bitstrings:
  - mean Z order
  - absolute Z order
  - nearest-neighbor/lattice ZZ correlations
- This is a hardware probe, not full HVK training.
