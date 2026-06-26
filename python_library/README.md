# HVK Python Package

This folder is the package-style version of the Hamiltonian Vision Kernel code.
It exists so the project can be imported from Python instead of only being run
through the research scripts in `Main/`, `Main2/`, and `Baselines/`.

The package mirrors the main research code closely. When changing core HVK
behavior, check whether the same update should be reflected here.

## Install

From this repository:

```powershell
python -m pip install -e .\python_library
```

With development tools:

```powershell
python -m pip install -e .\python_library[dev]
```

The package name is currently configured as `HVK`.

## Public API

Typical imports:

```python
from hvk import HVK2DConfig, run_hvk1d, run_hvk2d
```

Run HVK1D:

```python
outputs = run_hvk1d(
    "image.png",
    output_dir="hvk_outputs",
    image_size=256,
    patch_size=64,
    steps=120,
    model_variant="standard",
    device="auto",
)

print(outputs["history"]["total_loss"][-1])
```

Run the symmetric variant:

```python
outputs = run_hvk1d(
    "image.png",
    output_dir="hvk_outputs_symmetric",
    model_variant="symmetric",
    device="auto",
)
```

Run HVK2D:

```python
config = HVK2DConfig(
    image_path="image.png",
    output_dir="hvk2d_outputs",
    image_size=256,
    patch_size=64,
    steps=120,
    device="auto",
)

outputs = run_hvk2d(config)
print(outputs["phase_transition"])
```

## Device Behavior

The package follows the same device rules as the scripts:

- `device="auto"` uses CUDA for ordinary Torch work when available.
- HVK quantum circuits require PennyLane `lightning.gpu` for quantum CUDA.
- If `lightning.gpu` is missing, HVK quantum training uses CPU rather than
  creating mixed CPU/CUDA tensor failures.
- Circuit construction prefers `lightning.gpu`, then `lightning.qubit`, then
  `default.qubit`.

This is deliberate. Torch CUDA working does not mean PennyLane's quantum
simulator is also on CUDA.

## Configuration

Use `HVKRunConfig` for shared settings and `HVK2DConfig` for 2D runs. Important
fields include:

- `image_path`
- `output_dir`
- `image_size`
- `patch_size`
- `positional_dim`
- `steps`
- `lr`
- `device`
- `track_order_parameters`
- `save_outputs`
- `save_epoch_media`

Unsupported values should fail clearly instead of silently changing experiment
meaning.

## CLI

After install:

```powershell
hvk-run image.png --output-dir hvk_outputs --model-variant symmetric --steps 120
hvk-run image.png --variant-family hvk2d --output-dir hvk2d_outputs --steps 120
```

## Tests

From the repo root:

```powershell
python -m pytest python_library\tests
```

The tests cover public API behavior, bundled config loading, patching,
positional encoding, model variant selection, and order-parameter summaries.
