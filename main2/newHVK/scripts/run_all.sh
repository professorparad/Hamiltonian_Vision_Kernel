#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
.venv/bin/python main2/newHVK/run_newhvk_suite.py
cd main2/newHVK/paper_latex
pdflatex -interaction=nonstopmode newhvk_paper.tex
pdflatex -interaction=nonstopmode newhvk_paper.tex
