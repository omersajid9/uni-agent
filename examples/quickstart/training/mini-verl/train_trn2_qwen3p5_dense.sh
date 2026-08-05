#!/usr/bin/env bash
set -xeuo pipefail

project_name=${PROJECT_NAME:-"Uni-Agent-Qwen3.5-4B-trn2"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen3-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/data/uni_agent/swe_rebench_filtered_1150.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/data/uni_agent/swe_bench_verified.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"${RUNTIME_DIR}/data/uni_agent/runtime_env.yaml"}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}
# Must be launched from the repository root so Ray packages both `mini-verl/` and `uni_agent/`.
# --- Agent-framework rollout (replaces the swe_agent agent-loop) --------------
# Run-wide task base (agent + sandbox + sampling), loaded from this YAML by
# uni_agent.framework.task_runner.run_task and deep-merged onto each row's task.
# Same file-path idea as the old agent_loop_config_path; new (task-config) schema.
TASK_CONFIG=${TASK_CONFIG:-"examples/quickstart/training/task_config/react.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"hermes"}    # gateway tool-call parser; MUST match the model chat template
GATEWAY_COUNT=${GATEWAY_COUNT:-1}            # gateway actors fronting the engine
CONCURRENCY=${CONCURRENCY:-0}              # max in-flight rollout sessions (runner cap)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}

# DEVICE=cpu for the smoke path; DEVICE=auto picks Neuron when Trainium is present.
DEVICE=${DEVICE:-auto}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# Response length parameters
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 8))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 128))}

# Algorithm / batching
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-1}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-1}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
use_kl_loss=${USE_KL_LOSS:-true}
total_steps=${TOTAL_STEPS:-1}

# Parallelism: agentic path requires rollout TP/SP = 1 (LLM client drives one replica per batch)
# and colocate=false (turn-by-turn generation must not unshard FSDP around every turn).
rollout_cores=${ROLLOUT_CORES:-1}
actor_cores=${ACTOR_CORES:-1}
gen_batch_size=${GEN_BATCH_SIZE:-1}
ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE:-1}
weight_sync_channel=${WEIGHT_SYNC_CHANNEL:-host}

if [[ ! -f "${TASK_CONFIG}" ]]; then
  echo "ERROR: task config not found: ${TASK_CONFIG}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
# TransferQueue client is a no-op unless this is set in every process that touches TQ
# (driver + Ray actors that write trajectories).
export TRANSFER_QUEUE_ENABLE="${TRANSFER_QUEUE_ENABLE:-1}"

# ray job submit --no-wait --runtime-env $RUNTIME_ENV \

COLOCATE=${COLOCATE:-false}

if [ "${COLOCATE}" = "true" ]; then
    export MINI_VERL_DISABLE_EMPTY_CACHE="${MINI_VERL_DISABLE_EMPTY_CACHE:-true}"
fi

FSDP_STRATEGY=${FSDP_STRATEGY:-simple_fsdp}
LEFT_ALIGN=${LEFT_ALIGN:-true}
USE_PREFILL_FLASH=${USE_PREFILL_FLASH:-true}
PROJECT_NAME=${PROJECT_NAME:-mini_verl_grpo}

python -m mini_verl.run \
  device="${DEVICE}" \
  colocate="${COLOCATE}" \
  model.path="${MODEL_PATH}" \
  model.dtype=bfloat16 \
  model.attn_implementation="${ATTN_IMPL:-sdpa}" \
  data.train_files="${TRAIN_FILE}" \
  data.train_batch_size="${train_prompt_bsz}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.left_align_prompts="${LEFT_ALIGN}" \
  rollout.multi_turn.enable=true \
  rollout.agent_framework.task.task_name=swe_rebench \
  rollout.agent_framework.task.task_config_path="${TASK_CONFIG}" \
  rollout.agent_framework.gateway_count="${GATEWAY_COUNT}" \
  rollout.agent_framework.task.max_concurrent_sessions="${CONCURRENCY}" \
  rollout.agent_framework.max_batch_size="${gen_batch_size}" \
  rollout.agent_framework.batch_wait_s="${GEN_BATCH_WAIT_S:-0.05}" \
  rollout.agent_framework.log_dir="${AGENT_LOG_DIR}" \
  rollout.n="${n_resp_per_prompt}" \
  rollout.temperature="${temperature}" \
  rollout.top_p="${top_p}" \
  rollout.parallel.procs_per_node="${rollout_cores}" \
  rollout.parallel.tp_size=1 \
  actor.parallel.procs_per_node="${actor_cores}" \
  actor.parallel.tp_size=1 \
  actor.ppo_micro_batch_size="${ppo_micro_batch_size}" \
  actor.use_kl_loss="${use_kl_loss}" \
  trainer.total_training_steps="${total_steps}" \
  trainer.print_rollouts="${PRINT_ROLLOUTS:-2}" \
  trainer.logger="[${LOGGER:-console}]" \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  weight_sync.channel="${weight_sync_channel}" \
  "$@"



#   # Trainium
#   DEVICE=auto MODEL_DTYPE=bfloat16 ATTN_IMPL=sdpa \
#     ./examples/quickstart/training/train_arithmetic_miniverl.sh

