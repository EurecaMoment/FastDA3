#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$ROOT"
: "${CONDA_PREFIX:?Activate your Python 3.10 Conda environment first}"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python - <<'PY'
import os
from pathlib import Path
import platform
import sys

if platform.system() != "Linux" or platform.machine() != "x86_64":
    sys.exit("This installer supports Linux x86_64 with an NVIDIA GPU.")
libc, version = platform.libc_ver()
if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 31):
    sys.exit("The pinned Linux wheels require glibc >=2.31 (e.g. Ubuntu 22.04/24.04).")
if sys.version_info[:2] != (3, 10):
    sys.exit("Activate a Python 3.10 Conda environment (see environment.yml).")
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    sys.exit("python does not belong to the active Conda environment; check PATH.")
PY
if [[ -n ${FASTDA3_WHEELHOUSE:-} ]]; then
  # All serving/build dependencies must be present as Linux CPython 3.10 wheels.
  export PIP_NO_INDEX=1
  python -m pip install --no-index --find-links "$FASTDA3_WHEELHOUSE" -r requirements/fastda3.txt
else
  # Resolve common dependencies together below, with the pinned NumPy/Pillow versions.
  python -m pip install --no-deps torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url "${FASTDA3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  python -m pip install -r requirements/fastda3.txt
fi
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m fastda3.preflight --imports-only
printf '\nInstallation passed. Next: bash scripts/download.sh\n'
