#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# Asset preparation is the online phase; serving remains offline by default.
export HF_HUB_OFFLINE=0
exec "${PYTHON:-python}" "$FASTDA3_ROOT/scripts/download_assets.py" "$@"
