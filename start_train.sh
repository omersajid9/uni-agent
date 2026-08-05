#!/usr/bin/env bash
# Top-level training launcher. Selects the recipe by device:
#   bash start_train.sh            # DEVICE=neuron (default) -> mini-verl SWE smoke run
#   bash start_train.sh neuron     # explicit Trainium run
#   bash start_train.sh gpu        # verl Megatron SWE run
#   DEVICE=gpu bash start_train.sh # equivalent env-var form
set -euo pipefail

DEVICE="${1:-${DEVICE:-neuron}}"
export DEVICE

case "${DEVICE}" in
  neuron)
    N_RESP_PER_PROMPT=1 \
    ACTOR_CORES=2 \
    ROLLOUT_CORES=2 \
    TRAIN_PROMPT_BSZ=1 \
    PPO_MICRO_BATCH_SIZE=1 \
    CONCURRENCY=1 \
    WEIGHT_SYNC_CHANNEL=global_allreduce \
    TOTAL_STEPS=20 \
    bash examples/quickstart/training/mini-verl/train_swe_miniverl.sh
    ;;
  gpu)
    N_RESP_PER_PROMPT=1 \
    PPO_MINI_BATCH_SIZE=1 \
    MAX_RESPONSE_LENGTH=$((1024*4)) \
    RAY_ADDRESS=http://127.0.0.1:8265 \
    TASK_CONFIG=examples/quickstart/training/task_config/react.yaml \
    EXP_NAME=react_qwen3_4b_smoke \
    TOTAL_EPOCHS=1 \
    TOOL_PARSER=hermes \
    bash examples/quickstart/training/verl/train_qwen3p5_dense.sh
    ;;
  *)
    echo "ERROR: unknown DEVICE='${DEVICE}' (expected: neuron | gpu)" >&2
    exit 1
    ;;
esac
