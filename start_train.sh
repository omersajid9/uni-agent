#!/usr/bin/env bash

# DATA_DIR=$HOME/uni_agent_data \
# RUNTIME_DIR=$HOME/uni_agent_data \
# NNODES=1 NGPUS_PER_NODE=1 \
# TP=1 PP=1 CP=1 GEN_TP=1 \
# GATEWAY_COUNT=1 \
# CONCURRENCY=1 \
# TRAIN_PROMPT_BSZ=1 \
# N_RESP_PER_PROMPT=1 \
# PPO_MINI_BATCH_SIZE=1 \
# MAX_RESPONSE_LENGTH=$((1024*4)) \
# RAY_ADDRESS=http://127.0.0.1:8265 \
# TASK_CONFIG=examples/quickstart/training/task_config_react.yaml \
# EXP_NAME=react_qwen3_4b_smoke \
# TOTAL_EPOCHS=1 \
# TOOL_PARSER=hermes \
# bash examples/quickstart/training/train_qwen3p5_dense.sh

N_RESP_PER_PROMPT=1 \
ACTOR_CORES=2 \
ROLLOUT_CORES=2 \
TRAIN_PROMPT_BSZ=1 \
PPO_MICRO_BATCH_SIZE=1 \
CONCURRENCY=1 \
WEIGHT_SYNC_CHANNEL=global_allreduce \
TOTAL_STEPS=20 \
bash examples/quickstart/training/train_swe_miniverl.sh