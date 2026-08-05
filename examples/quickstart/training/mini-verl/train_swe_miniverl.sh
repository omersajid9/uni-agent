#!/usr/bin/env bash
# Trainium / CPU agentic GRPO on SWE-reBench (train) / SWE-Bench Verified (val) via mini-verl.
#
# This is the SWE twin of train_arithmetic_miniverl.sh

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

project_name=${PROJECT_NAME:-"Uni-Agent-swe-miniverl"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
# Run-wide task base (agent + sandbox + sampling), same schema as the GPU recipes.
TASK_CONFIG=${TASK_CONFIG:-"examples/quickstart/training/task_config/react.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"hermes"}

# Data: the preprocessed SWE parquet. Train on swe_rebench, validate on swe_bench_verified.
# Run setup_swe_data.sh once to materialize these.
DATA_DIR=${DATA_DIR:-"${HOME}/uni-agent-data/data/uni_agent"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/swe_rebench_filtered.parquet"}
# Which task family the parquet rows belong to — must match a `name` entry in TASK_CONFIG.
TASK_NAME=${TASK_NAME:-"swe_rebench"}

AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${REPO_ROOT}/logs/${project_name}/${exp_name}"}
mkdir -p "${AGENT_LOG_DIR}"

DEVICE=${DEVICE:-neuron}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
left_align_prompts=${LEFT_ALIGN:-true}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-4}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
use_kl_loss=${USE_KL_LOSS:-false}
total_steps=${TOTAL_STEPS:-20}

GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-2}
rollout_cores=${ROLLOUT_CORES:-1}
actor_cores=${ACTOR_CORES:-1}
gen_batch_size=${GEN_BATCH_SIZE:-1}
ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE:-1}
weight_sync_channel=${WEIGHT_SYNC_CHANNEL:-global_allreduce}

for f in "${TASK_CONFIG}" "${TRAIN_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: not found: ${f}" >&2
    echo "  (run setup_swe_data.sh from the repo root to build the parquet)" >&2
    exit 1
  fi
done

RUNTIME_ENV=${RUNTIME_ENV:-"examples/quickstart/training/runtime_env_swe_miniverl.yaml"}

export TOKENIZERS_PARALLELISM=false
export TRANSFER_QUEUE_ENABLE=1
export NEURON_CC_FLAGS="--model-type transformer"
export ACCELERATE_TORCH_DEVICE=neuron

ray job submit --no-wait --runtime-env "${RUNTIME_ENV}" \
    -- python3 -m mini_verl.run \
    device="${DEVICE}" \
    colocate=false \
    model.path="${MODEL_PATH}" \
    model.dtype="${MODEL_DTYPE}" \
    model.attn_implementation="${ATTN_IMPL}" \
    data.train_files="${TRAIN_FILE}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.left_align_prompts="${left_align_prompts}" \
    rollout.multi_turn.enable=true \
    rollout.agent_framework.task.task_name="${TASK_NAME}" \
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
    actor.log_prob_micro_batch_size="${ppo_micro_batch_size}" \
    actor.use_kl_loss="${use_kl_loss}" \
    actor.mem_snapshot=true \
    trainer.total_training_steps="${total_steps}" \
    trainer.print_rollouts="${PRINT_ROLLOUTS:-1}" \
    trainer.logger="${LOGGER:-[console,wandb]}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    weight_sync.channel="${weight_sync_channel}" \
    "$@"
