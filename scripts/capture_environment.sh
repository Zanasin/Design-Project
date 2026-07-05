#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

{
    echo "Capture date:"
    date --iso-8601=seconds
    echo

    echo "Operating system:"
    cat /etc/fedora-release
    echo

    echo "Kernel:"
    uname -r
    echo

    echo "Python:"
    python --version
    echo

    echo "Python path:"
    command -v python
    echo

    echo "Git:"
    git --version
    echo

    echo "GPU:"
    nvidia-smi \
      --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader
    echo

    echo "PyTorch:"
    python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_runtime:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("compute_capability:", torch.cuda.get_device_capability(0))
PY
    echo

    echo "Dependency check:"
    python -m pip check
} > environment_info.txt

echo "Saved environment information to environment_info.txt"
