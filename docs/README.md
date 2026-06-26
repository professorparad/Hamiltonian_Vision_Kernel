# Architecture Page

This folder contains the static documentation page for the repository. The page
is meant to give a quick visual overview of the HVK pipeline without requiring a
Python environment.

The main file is:

```text
docs/index.html
```

## What Belongs Here

Use this folder for lightweight project documentation:

- architecture overview
- diagrams
- short explanations of the pipeline
- static assets needed by the page

Do not put benchmark outputs or training artifacts here unless they are meant to
be part of the public documentation.

## GitHub Pages

The repo has a Pages workflow that can publish this folder as a static site.
Enable GitHub Pages from repository settings and choose GitHub Actions as the
source.

The page is intentionally static. It should be easy to inspect even when the
quantum ML dependencies are not installed.
