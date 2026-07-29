#!/usr/bin/env bash
# Trainium / CPU agentic GRPO on the arithmetic tool task.
#
# This is the uni-agent-side twin of train_qwen3p5_dense.sh for the mini-verl path:
# same place in the tree, same task-config idea, same "launch from repo root" rule.
# The difference is the trainer: GPU SWE uses verl + Megatron + vLLM; this uses
# mini-verl's RayGRPOTrainer with Uni-Agent's gateway as the rollout provider
# (AgenticRollout → MiniVerlLLMClient → Neuron/CPU generate_tokens).
#
# Must be launched from the repository root so `uni_agent` and `mini_verl` are
# both importable (setup_neuron_env.sh installs both editable into the Neuron venv).
#
#   # CPU smoke
#   DEVICE=cpu ./examples/quickstart/training/train_arithmetic_miniverl.sh
#
#   # Trainium
#   DEVICE=auto MODEL_DTYPE=bfloat16 ATTN_IMPL=sdpa \
#     ./examples/quickstart/training/train_arithmetic_miniverl.sh
#
set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

project_name=${PROJECT_NAME:-"Uni-Agent-arithmetic-miniverl"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}
# Run-wide task base (agent + sandbox + sampling), same schema as the GPU recipes.
TASK_CONFIG=${TASK_CONFIG:-"examples/quickstart/training/task_config_arithmetic.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"hermes"}   # Qwen chat-template <tool_call> envelope; mini-verl bridge default
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-0}          # 0 = unbounded; bind this when a task provisions real sandboxes

AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${REPO_ROOT}/logs/${project_name}/${exp_name}"}
mkdir -p "${AGENT_LOG_DIR}"

# DEVICE=cpu for the smoke path; DEVICE=auto picks Neuron when Trainium is present.
DEVICE=${DEVICE:-auto}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# Token frame. Trajectory (prompt + every assistant turn + every tool observation) must fit
# MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH. The task config's agent.model.max_tokens_per_turn is
# the fixed max_new_tokens every compiled decode graph is shaped to; max_total_tokens must fit
# in (frame - one turn) or AgenticRollout refuses at startup. On Neuron keep the sum a multiple
# of 512 if NKI flash paths are enabled.
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

# Algorithm / batching (kept small: this is the integration gate, not a SWE-scale run)
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-4}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
use_kl_loss=${USE_KL_LOSS:-true}
total_steps=${TOTAL_STEPS:-3}

# Parallelism: agentic path requires rollout TP/SP = 1 (LLM client drives one replica per batch)
# and colocate=false (turn-by-turn generation must not unshard FSDP around every turn).
rollout_cores=${ROLLOUT_CORES:-2}
actor_cores=${ACTOR_CORES:-1}
gen_batch_size=${GEN_BATCH_SIZE:-8}
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

python -m mini_verl.run \
  device="${DEVICE}" \
  colocate=false \
  model.path="${MODEL_PATH}" \
  model.dtype="${MODEL_DTYPE}" \
  model.attn_implementation="${ATTN_IMPL}" \
  data.train_batch_size="${train_prompt_bsz}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.left_align_prompts=false \
  agent.enable=true \
  agent.task_name=arithmetic \
  agent.task_config_path="${TASK_CONFIG}" \
  agent.gateway_count="${GATEWAY_COUNT}" \
  agent.max_concurrent_sessions="${CONCURRENCY}" \
  agent.max_batch_size="${gen_batch_size}" \
  agent.batch_wait_s="${GEN_BATCH_WAIT_S:-0.05}" \
  agent.dataset_size="${DATASET_SIZE:-512}" \
  agent.log_dir="${AGENT_LOG_DIR}" \
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
