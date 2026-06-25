# Hamiltonian Vision Kernel

Python package for Hamiltonian Vision Kernel image reconstruction.

Included variants:

- `run_hvk1d(..., model_variant="standard")`: learnable Heisenberg-style `XX`, `YY`, and `ZZ` couplings.
- `run_hvk1d(..., model_variant="symmetric")`: U(1)-symmetric `J * ZZ + K * (XX + YY)` couplings.
- `run_hvk2d(...)`: 2D grid HVK with lattice-correlation order parameters.

## Install

From PyPI:

```bash
pip install HVK
```

From this repository before publishing:

```bash
pip install ./python_library
```

For development:

```bash
pip install -e ./python_library[dev]
```

## Python Usage

```python
from hvk import HVK2DConfig, run_hvk1d, run_hvk2d

one_d_outputs = run_hvk1d(
    "image.png",
    output_dir="hvk_outputs",
    image_size=256,
    patch_size=64,
    steps=120,
    model_variant="symmetric",
)

two_d_outputs = run_hvk2d(
    HVK2DConfig(
        image_path="image.png",
        output_dir="hvk2d_outputs",
        image_size=256,
        patch_size=64,
        steps=120,
        device="cpu",
    )
)

print(one_d_outputs["history"]["total_loss"][-1])
print(two_d_outputs["phase_transition"])
```

## Configuration

Use `HVKRunConfig` for shared settings and `HVK2DConfig` for 2D runs.

Supported configuration fields include:

- `device`: `auto`, `cpu`, or `cuda`
- `quantum_device`: currently `default.qubit`
- `n_qubits`: currently `6`
- `n_layers`: currently `2`
- `image_mode`: currently `grayscale`
- `encoding`: currently `sinusoidal`
- `image_path`, `output_dir`, `image_size`, `patch_size`
- `steps`, `lr`, `positional_dim`
- `track_order_parameters`, `save_outputs`, `save_epoch_media`

Unsupported values are rejected clearly instead of silently doing the wrong thing.

## Custom HVK2D Order Parameter

```python
from hvk import HVK2DConfig, run_hvk2d

def my_order_parameter(observables, energies, previous_order):
    value = float(observables[:, :6].mean())
    previous = value if previous_order is None else previous_order
    return {
        "mean_energy": float(energies.mean()),
        "mean_order_parameter": value,
        "order_parameter_susceptibility": abs(value - previous),
    }

outputs = run_hvk2d(
    HVK2DConfig(
        image_path="image.png",
        order_parameter_fn=my_order_parameter,
    )
)
```

## CLI Usage

```bash
hvk-run image.png --output-dir hvk_outputs --model-variant symmetric --steps 120
hvk-run image.png --variant-family hvk2d --output-dir hvk2d_outputs --steps 120
```

## Tests

```bash
cd python_library
python -m pytest
```

The tests cover the public API, model-variant selection, bundled config loading,
configuration validation, patching, positional encoding, HVK1D order parameters,
and HVK2D lattice-correlation order parameters.
