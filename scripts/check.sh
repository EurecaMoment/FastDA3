#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "${PYTHON:-python}" -m fastda3.preflight "$@"
