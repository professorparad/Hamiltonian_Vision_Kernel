#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
.venv/bin/python Main2/newHVK/run_extended_validation.py
