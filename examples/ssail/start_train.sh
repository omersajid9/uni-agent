#!/usr/bin/env bash
# Top-level training launcher. Selects the recipe by device:
#   bash start_train.sh            # DEVICE=neuron (default) -> mini-verl SWE smoke run
#   bash start_train.sh neuron     # explicit Trainium run
#   bash start_train.sh gpu        # verl Megatron SWE run
#   DEVICE=gpu bash start_train.sh # equivalent env-var form
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
[[ -f "${SCRIPT_DIR}/.env" ]] && { set -a; source "${SCRIPT_DIR}/.env"; set +a; }

DEVICE="${1:-${DEVICE:-neuron}}"
export DEVICE

case "${DEVICE}" in
  neuron)
    COLOCATE=true \
    TP_SIZE=2 \
    ACTOR_SP_SIZE=2 \
    FSDP_STRATEGY=simple_fsdp \
    ACTOR_CORES=4 \
    ROLLOUT_CORES=4 \
    MAX_PROMPT_LENGTH=4096 \
    MAX_RESPONSE_LENGTH=4096 \
    TRAIN_PROMPT_BSZ=8 \
    N_RESP_PER_PROMPT=4 \
    GEN_BATCH_SIZE=4 \
    TOTAL_STEPS=10 \
    bash "${SCRIPT_DIR}/neuron/train_qwen3p5_dense.sh"
    ;;
  gpu)
    N_RESP_PER_PROMPT=1 \
    PPO_MINI_BATCH_SIZE=1 \
    MAX_RESPONSE_LENGTH=$((1024*4)) \
    RAY_ADDRESS=http://127.0.0.1:8265 \
    TASK_CONFIG=examples/ssail/task_config/react.yaml \
    EXP_NAME=react_qwen3_4b_smoke \
    TOTAL_EPOCHS=1 \
    TOOL_PARSER=hermes \
    bash "${SCRIPT_DIR}/gpu/train_qwen3p5_dense.sh"
    ;;
  *)
    echo "ERROR: unknown DEVICE='${DEVICE}' (expected: neuron | gpu)" >&2
    exit 1
    ;;
esac