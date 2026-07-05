#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Python 3.12 is required but was not found."
    exit 1
fi

if [[ ! -d ".venv" ]]; then
    python3.12 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt
python -m pip check

echo
echo "Environment setup completed."
echo "Activate it with:"
echo "source .venv/bin/activate"
