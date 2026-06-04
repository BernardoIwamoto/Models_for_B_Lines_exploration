#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

python - <<'PY'
import sys

print(f"Python: {sys.version}")
print("Environment bootstrap complete.")
PY
