#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
.venv/bin/python Main2/newHVK/run_newhvk_suite.py --q1-validation --write-q1-report
cd Main2/newHVK/paper_latex
pdflatex -interaction=nonstopmode newhvk_q1_validation_report.tex
pdflatex -interaction=nonstopmode newhvk_q1_validation_report.tex
