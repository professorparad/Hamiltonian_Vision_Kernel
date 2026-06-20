# CIFAR-10 Benchmark Workspace

This folder separates the CIFAR-10 subset benchmark by model so each experiment
can be run independently and inspected before adding numbers to the LaTeX table.

Dataset expected by default:

```text
Baselines/datasets/cifar10_subset/images/
```

Each runner uses 200 epochs by default and writes:

- per-image model outputs
- `per_image_metrics.csv`
- `aggregate_metrics.csv`
- `aggregate_metrics.json`
- `metrics_summary.png`

## Run 1D HVK

```bash
.venv/bin/python Baselines/cifar10_benchmark/hvk1d/run_hvk1d_cifar10.py
```

Outputs:

```text
Baselines/cifar10_benchmark/hvk1d/outputs/
```

## Run 2D HVK

```bash
.venv/bin/python Baselines/cifar10_benchmark/hvk2d/run_hvk2d_cifar10.py
```

Outputs:

```text
Baselines/cifar10_benchmark/hvk2d/outputs/
```

## Run GAN

```bash
.venv/bin/python Baselines/cifar10_benchmark/gan/run_gan_cifar10.py
```

Outputs:

```text
Baselines/cifar10_benchmark/gan/outputs/
```

## Notes

The runners are resumable. If an image has a `.done` file in its output folder,
that image is skipped on the next run. Delete the corresponding image output
folder if you want to rerun it from scratch.
