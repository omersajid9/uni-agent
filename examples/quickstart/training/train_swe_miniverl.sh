#!/usr/bin/env bash
# Trainium / CPU agentic GRPO on SWE-reBench (train) / SWE-Bench Verified (val) via mini-verl.
#
# This is the SWE twin of train_arithmetic_miniverl.sh: same mini-verl RayGRPOTrainer + Uni-Agent
# gateway rollout, same "launch from repo root" rule, same colocate=false / rollout TP=1
# constraints. The difference is the domain: real SWE problems, a Modal sandbox running the SWE
# test harness, and a trajectory-level reward (tests resolved) posted back by the task.
#
# Uni-Agent owns all the SWE + sandbox logic (task, agent loop, Modal sandbox, reward); mini-verl
# is just the rollout (generation) + actor (GRPO update). The glue is:
#   * data.train_files  -- the preprocessed SWE parquet (run prepare_swe_data.sh first)
#   * agent.task_name   -- swe_rebench (train) or swe_bench (val)
#   * TASK_CONFIG       -- task_config_swe_miniverl.yaml (sandbox provider + agent + token budgets)
#
# Must be launched from the repository root so `uni_agent` and `mini_verl` are both importable
# (setup_neuron_env.sh installs both editable into the Neuron venv).
#
#   # 0. prepare data once (small smoke set):
#   bash examples/quickstart/training/prepare_swe_data.sh
#
#   # 1. CPU smoke (validates the loop without spending Trainium time):
#   DEVICE= DEVICE=cpu \
#     bash examples/quickstart/training/train_swe_miniverl.sh
#
#   # 2. Trainium:
#   DEVICE=neuron MODEL_DTYPE=bfloat16 ATTN_IMPL=sdpa \
#     bash examples/quickstart/training/train_swe_miniverl.sh
#
# Prereqs beyond the arithmetic run: `modal` (pip install modal; modal token new) and `swebench`
# (pip install swebench) so the sandbox can pull the per-instance SWE image and grade tests. Set
# MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment (or in runtime_env_miniverl.yaml).
set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

project_name=${PROJECT_NAME:-"Uni-Agent-swe-miniverl"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
# Run-wide task base (sandbox provider + agent loop + token budgets). Neuron-scaled twin of the
# GPU recipe's task_config_react.yaml — adds the max_tokens_per_turn mini-verl requires.
TASK_CONFIG=${TASK_CONFIG:-"examples/quickstart/training/task_config_swe_miniverl.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"hermes"}   # Qwen chat-template <tool_call> envelope; mini-verl bridge default

# Data: the preprocessed SWE parquet. Train on swe_rebench, validate on swe_bench_verified.
# Run prepare_swe_data.sh once to materialize these.
DATA_DIR=${DATA_DIR:-"${HOME}/data/uni_agent"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/swe_rebench_filtered.parquet"}
# Which task family the parquet rows belong to — must match a `name` entry in TASK_CONFIG.
TASK_NAME=${TASK_NAME:-"swe_rebench"}

AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${REPO_ROOT}/logs/${project_name}/${exp_name}"}
mkdir -p "${AGENT_LOG_DIR}"

# DEVICE=cpu for the smoke path; DEVICE=neuron forces Trainium (auto-detect fails inside a
# `ray job submit` driver because torch.neuron.is_available() probes NRT with an empty
# NEURON_RT_VISIBLE_CORES and falls back to cpu). The driver's resolved device is propagated
# to every Ray worker via MINI_VERL_DEVICE, so it must be neuron on Trainium.
DEVICE=${DEVICE:-neuron}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# Token frame. Trajectory must fit MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH. AgenticRollout context =
# frame - max_tokens_per_turn, and max_total_tokens must fit inside that.
#
# Memory tradeoff on Trainium: actor.compute_log_prob HBM scales with (seq × batch). Prefer a
# LONGER frame for SWE (prompts are already ~2k) and a SMALLER parallel batch — not the reverse.
# Defaults: 4096+4096=8192 frame with TRAIN_PROMPT_BSZ=1 and N_RESP=2 so log-prob only sees 2
# sequences. The prior 12k + bsz=2×n=2 OOMed; the arithmetic-sized 1.5k frame rejected every SWE
# prompt at step 1.
#
# LEFT_ALIGN routes the actor forward through the Neuron NKI flash kernel (fwd+bwd) by dropping the
# attention_mask, which cuts the backward scratchpad from O(B·S²) (the dense masked path that OOMs
# at 8k context on a 24 GiB logical core) to a flat tiled ~0.5-1 GiB. The agentic collate now emits
# the prompt_length field and repacks input_ids to the contiguous [prompt|response|PAD] layout the
# flash path requires (mirrors the single-turn RolloutGenMixin._left_align_batch). REQUIRES
# (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) % 512 == 0 (default 4096+4096=8192 ✓; asserted on
# Neuron). prompts/responses/attention_mask stay in verl layout so _print_rollouts is unaffected.
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
left_align_prompts=${LEFT_ALIGN:-true}

# Algorithm / batching: shrink parallel width to pay for the longer SWE frame.
# GRPO still needs n>=2; keep prompts/step at 1 so the actor log-prob / update batch stays tiny.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-4}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
use_kl_loss=${USE_KL_LOSS:-false}
total_steps=${TOTAL_STEPS:-20}

# Parallelism: agentic path requires rollout TP/SP = 1 and colocate=false.
# CONCURRENCY / GEN_BATCH_SIZE bound in-flight Modal sessions and decode coalescing — keep them
# near train_prompt_bsz × n so you don't pile up sandboxes/KV while the actor is still one-step.
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
    echo "  (run examples/quickstart/training/prepare_swe_data.sh to build the parquet)" >&2
    exit 1
  fi
done

RUNTIME_ENV=${RUNTIME_ENV:-"examples/quickstart/training/runtime_env_swe_miniverl.yaml"}

# These four are re-emitted by mini_verl's own ray.init() (see ray_trainer.py), so they must NOT
# also appear in the job runtime_env yaml (that triggers a merge conflict). Exporting them here
# puts them in the driver's os.environ, which mini_verl forwards to the Ray workers itself.
# (PYTHONUNBUFFERED / RAY_DEDUP_LOGS / TQ_NUM_THREADS live in the yaml instead.)
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
    agent.enable=true \
    agent.task_name="${TASK_NAME}" \
    agent.task_config_path="${TASK_CONFIG}" \
    agent.gateway_count="${GATEWAY_COUNT}" \
    agent.max_concurrent_sessions="${CONCURRENCY}" \
    agent.max_batch_size="${gen_batch_size}" \
    agent.batch_wait_s="${GEN_BATCH_WAIT_S:-0.05}" \
    agent.log_dir="${AGENT_LOG_DIR}" \
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
