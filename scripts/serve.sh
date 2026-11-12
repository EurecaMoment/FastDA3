#!/usr/bin/env bash
set -euo pipefail
FASTDA3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$FASTDA3_ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${DA3_API_KEY:?Set DA3_API_KEY in the environment or .env}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
exec "${PYTHON:-python}" -u -m fastda3.server "$@"
