# Package Tests

These tests cover the installable `hvk` package rather than the research script
entry points.

They check:

- public API imports
- config validation
- HVK1D and HVK2D run wiring
- order-parameter summaries
- package data loading

Run from the repo root:

```powershell
python -m pytest python_library\tests
```

If the package import fails, install it in editable mode:

```powershell
python -m pip install -e .\python_library
```
