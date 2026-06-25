# Baseline Algorithms

The active baseline suites live in `cifar10_comparisons/` and
`monalisa_comparisons/`.

Run the full CIFAR-10 32x32 comparison from one command:

```bash
python Baselines/cifar10_comparisons/main.py --count 5 --epochs 200 --device cpu
```

Run selected methods:

```bash
python Baselines/cifar10_comparisons/main.py \
  --methods hvk1d hvk2d symmetric gan phl \
  --count 3 \
  --epochs 200 \
  --device cpu \
  --skip-download
```

Included methods:

- `hvk1d`, `hvk2d`, and `symmetric` quantum/HVK runners.
- `gan`, `mlp`, `cnn`, and `autoencoder` reconstruction baselines.
- `phl` Parameterized Hamiltonian Learning segmentation baseline.

Combined outputs are written to `Baselines/cifar10_comparisons/outputs/`.

Run the same layout on Mona Lisa:

```bash
python Baselines/monalisa_comparisons/main.py --methods mlp phl --epochs 200 --device cpu
```

Each suite writes:

- one combined per-image CSV
- one aggregate metrics CSV and JSON
- one aggregate metric comparison PNG
- `visuals/<method>/` for plots and graphics
- `per_method_metrics/<method>/` for each method's CSV/JSON files
