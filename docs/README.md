# GitHub Pages Site

This folder contains the static GitHub Pages deployment for the Hamiltonian
Vision Kernel repository. It is intentionally plain HTML/CSS with no build step,
no external JavaScript, and no placeholder links.

The deployed entry point is:

```text
docs/index.html
```

## Public Assets

The site links only to files committed under `docs/assets`:

- `assets/papers/paper_hvk.pdf`
- `assets/papers/hvk_methodology_outputs.pdf`
- `assets/data/cifar32_aggregate_metrics.csv`
- `assets/data/cifar32_per_image_metrics.csv`
- `assets/data/cifar32_aggregate_metrics.json`
- `assets/figures/*.png`

If the paper or figures are regenerated, copy the final public versions into
this folder before deploying.

## GitHub Pages

The Pages workflow publishes this folder directly:

```text
.github/workflows/pages.yml
```

Repository settings should use GitHub Actions as the Pages source. The workflow
validates that `docs/index.html`, `docs/.nojekyll`, the paper PDF, and the main
CIFAR benchmark figure exist before uploading the artifact.
