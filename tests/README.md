# Tests

This folder contains the automated checks for the repo. The tests are small on
purpose; they protect the parts that are easiest to break while refactoring:

- decoder shape and value behavior
- preprocessing helpers
- training smoke paths
- CIFAR benchmark helper functions
- MPS utility helpers
- package import paths
- retained numerical claim-to-artifact mappings, including the corrected TOST input

The GitHub test suite does not depend on local files in `overleaf_docs/`.
Manuscript sources and review notes are maintained separately from the code.

## Run

From the repo root:

```powershell
python -m pytest tests
```

To include the package tests:

```powershell
python -m pytest tests python_library\tests
```

If `pytest` is not installed:

```powershell
python -m pip install pytest
```

## Notes

The tests are not meant to validate the full research claim. They are guardrails
for basic behavior. Full experiment validation still comes from benchmark runs,
metric CSVs, output images, and order-parameter plots.
