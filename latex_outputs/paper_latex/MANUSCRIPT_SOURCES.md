# HVK manuscript source of truth

The publication is maintained as exactly two authoritative LaTeX documents:

1. `paper_hvk.tex` - the Q1-journal main manuscript. Its central contribution is the Hamiltonian Vision Kernel (HVK) as a new framework for hybrid quantum-classical visual representation. Quantum-art and reconstruction outputs are applications of HVK, not a separate competing project.
2. `supplementary_study.tex` - the companion supplementary material containing resource-matched ablations, leakage audits, extended controls, statistical detail, and hardware-reproduction methodology.

The duplicate mirror copies (`experiments/manuscript_skeleton.tex`, `project_artifacts/manuscript_skeleton.tex`) and stale independent drafts (`latex_outputs/hvk_methodology_outputs.tex`, `latex_outputs/papers/newhvk_paper.tex`) have been removed to keep a single source of truth; they remain in git history if needed.

The one exception is `Main2/newHVK/paper_latex/newhvk_q1_validation_report.tex` — a separate addendum document actively wired into that sub-project's own build scripts (`run_newhvk_suite.py`, `scripts/run_q1_validation_suite.sh`). It is not part of this manuscript and should be left alone.

Recommended build commands, run from this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper_hvk.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_hvk.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_study.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_study.tex
```
