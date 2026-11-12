#!/usr/bin/env bash
# Sourced by the Bash entry points. Exported caller settings override .env.
fastda3_load_environment() {
  local root config name
  local -A saved=()
  root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  export FASTDA3_ROOT="$root"
  config=${FASTDA3_ENV_FILE:-"$root/.env"}
  if [[ -n ${FASTDA3_ENV_FILE:-} && ! -f $config ]]; then
    printf 'Configuration file does not exist: %s\n' "$config" >&2
    return 1
  fi
  if [[ -f $config ]]; then
    while IFS= read -r name; do
      saved["$name"]=${!name-}
    done < <(compgen -e)
    set -a
    source "$config"
    set +a
    for name in "${!saved[@]}"; do
      if [[ ${!name-} != "${saved[$name]}" ]]; then
        export "$name=${saved[$name]}"
      fi
    done
  fi
  export FASTDA3_ROOT="$root"
  cd -- "$root"
}
fastda3_load_environment
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
