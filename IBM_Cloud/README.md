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
- `run_hardware_latent_validation.py`: compares exact statevector latent
  observables with hardware-measured latent observables for the same patch
  circuits.
- `run_ibm_epoch_probe.py`: submits epoch-labeled patch circuits to inspect how
  the proxy changes across selected epoch labels.
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

The scripts also support environment variables. This is the safest workflow
because tokens never enter source code:

```powershell
$env:IBM_QUANTUM_TOKEN="paste-token-here"
```

For permanent local use on Windows, set it as a user environment variable:

```powershell
[Environment]::SetEnvironmentVariable("IBM_QUANTUM_TOKEN", "paste-token-here", "User")
```

Restart the terminal after setting a permanent variable. Never commit `.env`
files, screenshots, notebooks, or logs containing tokens.

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

## Next Best Experiment: Hardware Latent Validation

The most defensible next experiment is not full 200-epoch QPU training. It is a
latent replacement/validation run:

1. Build the same HVK patch circuits used by the simulator.
2. Compute exact statevector order and correlation metrics locally.
3. Submit the same circuits to IBM Heron.
4. Compare simulator latent observables against hardware latent observables.

This directly tests whether the QPU preserves the HVK latent structure:

- $\langle M_z\rangle$
- $|\langle M_z\rangle|$
- $\langle ZZ\rangle$
- proxy loss
- hardware-minus-simulator deltas

Dry-run the simulator side first:

```powershell
python IBM_Cloud\run_hardware_latent_validation.py --dataset IBM_Cloud\datasets\monalisa_patches.npz --variant both --max-patches 1 --shots 100 --dry-run
```

Submit a small Heron validation job:

```powershell
python IBM_Cloud\run_hardware_latent_validation.py --dataset IBM_Cloud\datasets\monalisa_patches.npz --variant both --backend ibm_fez --n-qubits 6 --shots 100 --max-patches 1
```

For a stronger but still controlled run:

```powershell
python IBM_Cloud\run_hardware_latent_validation.py --dataset IBM_Cloud\datasets\monalisa_patches.npz --variant hvk2d --backend ibm_fez --n-qubits 6 --shots 100 --max-patches 8
```

Outputs:

- `IBM_Cloud/outputs/hardware_latent_validation.json`
- `IBM_Cloud/outputs/hardware_latent_validation.csv`
- `IBM_Cloud/outputs/hardware_latent_validation.png`

Paper-safe claim:

> We compare exact statevector HVK latent observables with IBM Heron
> measurements for identical image-patch circuits. This validates hardware
> preservation of the Hamiltonian latent signal without claiming full
> hardware-executed image reconstruction.

## Cross-Quantum-Computer Validation

Use `run_cross_quantum_validation.py` when you want the same HVK latent circuits
compared across local statevector, IBM, IonQ, AWS Braket, or Azure Quantum.

Install optional cloud provider SDKs:

```powershell
python -m pip install -r IBM_Cloud\requirements-cross-cloud.txt
```

Local exact statevector baseline:

```powershell
python IBM_Cloud\run_cross_quantum_validation.py --provider statevector --variant both --max-patches 1
```

IBM Quantum:

```powershell
$env:IBM_QUANTUM_TOKEN="paste-token-here"
python IBM_Cloud\run_cross_quantum_validation.py --provider ibm --backend ibm_fez --variant both --shots 100 --max-patches 1
```

IonQ direct cloud:

```powershell
$env:IONQ_API_TOKEN="paste-token-here"
python IBM_Cloud\run_cross_quantum_validation.py --provider ionq --backend ionq_simulator --variant both --shots 100 --max-patches 1
```

AWS Braket:

```powershell
aws configure
python IBM_Cloud\run_cross_quantum_validation.py --provider braket --backend arn:aws:braket:::device/quantum-simulator/amazon/sv1 --variant both --shots 100 --max-patches 1
```

Azure Quantum:

```powershell
az login
$env:AZURE_QUANTUM_RESOURCE_ID="/subscriptions/<sub>/resourceGroups/<group>/providers/Microsoft.Quantum/Workspaces/<workspace>"
$env:AZURE_QUANTUM_LOCATION="<azure-region>"
python IBM_Cloud\run_cross_quantum_validation.py --provider azure --backend <target-name> --variant both --shots 100 --max-patches 1
```

Azure target names depend on the providers enabled in your Azure Quantum
workspace. List targets from the Azure portal, VS Code QDK extension, or Azure
Quantum Python tools, then pass the target name to `--backend`.

The output files are provider-specific:

- `IBM_Cloud/outputs/cross_quantum_validation_statevector.json`
- `IBM_Cloud/outputs/cross_quantum_validation_ibm.json`
- `IBM_Cloud/outputs/cross_quantum_validation_ionq.json`
- `IBM_Cloud/outputs/cross_quantum_validation_braket.json`
- `IBM_Cloud/outputs/cross_quantum_validation_azure.json`

Paper-safe claim:

> We evaluate identical HVK latent circuits across simulator and cloud quantum
> backends, reporting deviations in order, nearest-neighbor correlation, and
> proxy loss. These are latent validation measurements, not full hardware image
> reconstructions.

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
