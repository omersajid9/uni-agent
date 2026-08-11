#!/usr/bin/env bash
# Top-level environment bootstrapper. Selects the setup script by device:
#   bash setup_env.sh              # DEVICE=neuron (default) -> scripts/setup_env_trn.sh
#   bash setup_env.sh neuron       # explicit Trainium setup
#   bash setup_env.sh gpu          # scripts/setup_env_gpu.sh (phase0..phase3 + verify, default "all")
#   bash setup_env.sh gpu phase1   # extra args after the device pass straight through
#   DEVICE=gpu bash setup_env.sh   # equivalent env-var form
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
[[ -f "${SCRIPT_DIR}/.env" ]] && { set -a; source "${SCRIPT_DIR}/.env"; set +a; }

DEVICE="${1:-${DEVICE:-neuron}}"
[[ $# -gt 0 ]] && shift
export DEVICE

case "${DEVICE}" in
  neuron)
    bash "$SCRIPT_DIR/scripts/setup_env_trn.sh" "$@"
    ;;
  gpu)
    bash "$SCRIPT_DIR/scripts/setup_env_gpu.sh" "$@"
    ;;
  *)
    echo "ERROR: unknown DEVICE='${DEVICE}' (expected: neuron | gpu)" >&2
    exit 1
    ;;
esac
