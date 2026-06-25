# Mona Lisa Comparisons

This folder runs the same baseline suite used by `cifar10_comparisons/` on
`Main/data/monalisa.jpg`.

Run all methods:

```bash
python Baselines/monalisa_comparisons/main.py --epochs 200 --device cpu
```

Run selected methods:

```bash
python Baselines/monalisa_comparisons/main.py \
  --methods hvk1d hvk2d symmetric gan phl \
  --epochs 200 \
  --device cpu
```

Outputs use the same layout as CIFAR:

- `outputs/monalisa_per_image_metrics.csv`
- `outputs/monalisa_aggregate_metrics.csv`
- `outputs/monalisa_aggregate_metrics.json`
- `outputs/monalisa_metric_comparison.png`
- `outputs/visuals/<method>/`
- `outputs/per_method_metrics/<method>/`
