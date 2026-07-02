# python_library/ — The Installable HVK Package

## What It Is

`python_library/` is a **pip-installable Python package** named `HVK` (package namespace: `hvk`).  
It is the **packaged, clean-API version** of the research code — essentially a mirror of `Main/` and `Main2/` organized as a proper Python library with a public API, config dataclasses, and a CLI entry point.

Install it with:
```bash
pip install -e python_library/
```

After install, the CLI command `hvk-run` becomes available system-wide.

---

## Relationship to Main/ and Main2/

| `python_library/src/hvk/` | Mirrors |
|---------------------------|---------|
| `preprocessing/` | `Main/src/preprocessing/` |
| `tensornetworks/` | `Main/src/tensornetworks/` |
| `quantum/` | `Main/src/quantum/` |
| `decoder/` | `Main/src/decoder/` |
| `reconstruction/` | `Main/src/reconstruction/` |
| `training/` | `Main/src/training/` |
| `visualization/` | `Main/src/visualization/` |
| `hvk2d/` | `Main2/src/` |

**The library is a stable snapshot** — research experiments in `Main/` and `Main2/` evolve faster than the library. When the student makes a fix in `Main/`, the fix may need to be ported to `python_library/` separately.

---

## Public API (`src/hvk/api.py`)

Two main functions exposed:

### `run_hvk1d(image_path, *, model_variant="standard", steps=120, ...)`
- Validates config via `HVKRunConfig.validate()`
- Calls `train()` from `hvk.training.training`
- Returns the full `outputs` dict (same as `Main/` training)

### `run_hvk2d(config=None, **overrides)`
- Delegates to `hvk.hvk2d.training.run_hvk2d()`
- Accepts `HVK2DConfig` or keyword overrides

### CLI entry point
```bash
hvk-run <image_path> [--model-variant standard|symmetric] [--variant-family hvk1d|hvk2d] [--steps N] ...
```

---

## Config Dataclasses (`src/hvk/config.py`)

### `HVKRunConfig` (frozen dataclass)
All parameters for HVK1D training. Key validation rules enforced by `.validate()`:
- `device` must be `auto | cpu | cuda`
- `n_qubits` must be 6 (hardcoded circuit)
- `n_layers` must be 2
- `image_mode` must be `grayscale`
- `encoding` must be `sinusoidal`
- `image_size % patch_size == 0`

### `HVK2DConfig(HVKRunConfig)`
Inherits all above, adds:
- `lr = 0.004` (different default from HVK1D's 0.003)
- `steps = 200`
- `save_gif = True`
- `order_parameter_fn` — optional custom order parameter function hook

---

## Package Metadata (`pyproject.toml`)

| Field | Value |
|-------|-------|
| Package name | `HVK` |
| Version | `0.1.0` |
| Python | >=3.10 |
| License | MIT |
| CLI script | `hvk-run` → `hvk.api:main` |
| Test runner | pytest (testpaths = `tests/`) |
| Linter | ruff |

---

## Sync Status (verified 2026-07-02)

**Currently in sync.** Git history shows the library was created as a bulk copy of `Main/` (commit `9d8ba12`) and subsequent fixes have been applied to both simultaneously (commits `ecd6b6e "auto sync"` and `e7016d9` for the CUDA device fallback). There is no divergence at HEAD.

The library adds things `Main/` does not have:
- `api.py` — clean public functions `run_hvk1d()` / `run_hvk2d()`
- `config.py` — `HVKRunConfig` and `HVK2DConfig` dataclasses with validation
- `pyproject.toml` — makes it pip-installable as `hvk-run` CLI

If the student makes changes to `Main/` without syncing, the library will fall behind. Check with `git diff HEAD -- Main/src/ python_library/src/hvk/` to detect drift.

*Last updated: 2026-07-02*
