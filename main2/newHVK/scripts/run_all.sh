#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
.venv/bin/python main2/newHVK/run_newhvk_suite.py --full-suite --q1-validation --write-q1-report
if [ -f main2/newHVK/paper_latex/newhvk_paper.tex ]; then
  (
    cd main2/newHVK/paper_latex
    pdflatex -interaction=nonstopmode newhvk_paper.tex
    pdflatex -interaction=nonstopmode newhvk_paper.tex
  )
fi
if [ -f main2/newHVK/paper_latex/newhvk_q1_validation_report.tex ]; then
  (
    cd main2/newHVK/paper_latex
    pdflatex -interaction=nonstopmode newhvk_q1_validation_report.tex
    pdflatex -interaction=nonstopmode newhvk_q1_validation_report.tex
  )
fi
