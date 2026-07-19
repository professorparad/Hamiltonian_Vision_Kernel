# HVK manuscript source of truth

The publication is maintained as exactly three authoritative LaTeX documents:

1. `paper_hvk.tex` - the Q1-journal main manuscript. Its central contribution is the Hamiltonian Vision Kernel (HVK) as a new framework for hybrid quantum-classical visual representation. Quantum-art and reconstruction outputs are applications of HVK, not a separate competing project.
2. `supplementary_study.tex` - the companion supplementary material containing resource-matched ablations, leakage audits, extended controls, statistical detail, and hardware-reproduction methodology.
3. `literature_review.tex` - a standalone, arXiv-citable literature review ("Hybrid Quantum--Classical Machine Learning for Computer Vision: A Mathematical Literature Review") surveying the broader field with explicit mathematical formalism (VQAs, barren plateaus, QCNNs, tensor networks, quantum autoencoders, symmetry/equivariance, Hamiltonian-based regularization, dequantization). Its ~66 references were each independently verified via web search against their real venue/arXiv ID before inclusion — none were reproduced from an unverified secondary source. It is independent of, and does not need to be cited by, the other two documents.

The duplicate mirror copies (`experiments/manuscript_skeleton.tex`, `project_artifacts/manuscript_skeleton.tex`) and stale independent drafts (`latex_outputs/hvk_methodology_outputs.tex`, `latex_outputs/papers/newhvk_paper.tex`) have been removed to keep a single source of truth; they remain in git history if needed.

The one exception is `Main2/newHVK/paper_latex/newhvk_q1_validation_report.tex` — a separate addendum document actively wired into that sub-project's own build scripts (`run_newhvk_suite.py`, `scripts/run_q1_validation_suite.sh`). It is not part of this manuscript and should be left alone.

Recommended build commands, run from this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../.. paper_hvk.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../.. paper_hvk.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_study.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_study.tex
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex
pdflatex -interaction=nonstopmode -halt-on-error literature_review.tex
```
