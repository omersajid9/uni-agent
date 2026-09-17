#!/usr/bin/env bash
# Single-node (1x8 H100) SWE RL for Qwen3-4B with TaskPilot band filtering.
#
# One invocation = one iteration (a ~200-update climb). Between iterations, point
# MODEL_PATH at the previous iteration's exported HF checkpoint and bump ITERATION:
# policy weights carry over while optimizer and RNG state reset. See README.md.
#
#   ITERATION=1 MODEL_PATH=~/models/Qwen3-4B \
#       TRAIN_FILE=~/data/frognano/iter1/accepted.parquet \
#       examples/frognano/train_frognano_qwen3_4b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

: "${MODEL_PATH:=${HOME}/models/Qwen3-4B}"
: "${TRAIN_FILE:=${HOME}/data/frognano/swe_rebench_pool.parquet}"
: "${VAL_FILE:=${HOME}/data/frognano/swe_bench_verified.parquet}"
: "${PYTHON_BIN:=python3}"
: "${RAY_BIN:=ray}"

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

TASK_CONFIG="${TASK_CONFIG:-examples/frognano/task_config_leaf.yaml}"
TASKPILOT_CONFIG="${TASKPILOT_CONFIG:-examples/frognano/configs/taskpilot.yaml}"
ITERATION="${ITERATION:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"   # must match the model's chat template

PROJECT_NAME="${PROJECT_NAME:-frognano}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-frognano_iter${ITERATION}_$(date +%Y%m%d_%H%M)}"
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${REPO_ROOT}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

# separate_async splits the node into disjoint pools: 2 GPUs train (Ulysses SP 2),
# 6 single-GPU rollout engines keep the 8-trajectory groups flowing.
TRAINER_NNODES="${TRAINER_NNODES:-1}"
TRAINER_GPUS_PER_NODE="${TRAINER_GPUS_PER_NODE:-2}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-6}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ULYSSES_SP="${ULYSSES_SP:-${TRAINER_GPUS_PER_NODE}}"

# 32 task groups x 8 trajectories = 256 trajectories per update.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))}"
# Band filtering discards groups mid-step, so keep spare prompts in flight to refill them.
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-57344}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN / ULYSSES_SP))}"

# reward.custom_reward_function is deliberately left unset. Setting it would route
# scoring through a RewardLoopWorker, which re-derives the score from the raw
# TaskResult; unset, the framework takes reward_source=agent_runner and the length
# penalty below stays authoritative over rm_scores. Either way reward_metrics keeps
# the raw acc, which is what TaskPilot bands on.

# Success-gated log-length penalty; the paper turns it on from iteration 3.
LENGTH_PENALTY_ENABLE="${LENGTH_PENALTY_ENABLE:-$([[ ${ITERATION} -ge 3 ]] && echo True || echo False)}"
LENGTH_PENALTY_FREE_TOKENS="${LENGTH_PENALTY_FREE_TOKENS:-24576}"
LENGTH_PENALTY_ALPHA="${LENGTH_PENALTY_ALPHA:-0.1}"

GATEWAY_COUNT="${GATEWAY_COUNT:-4}"
CONCURRENCY="${CONCURRENCY:-128}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

if ! "${RAY_BIN}" status >/dev/null 2>&1; then
    echo "Starting a local Ray cluster on 8 GPUs..."
    "${RAY_BIN}" start --head --num-gpus=8
fi

"${RAY_BIN}" job submit --no-wait \
    --working-dir="${REPO_ROOT}" \
    --runtime-env-json="{\"env_vars\": {\"RAY_DEDUP_LOGS\": \"0\"}}" \
    -- "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    trainer.v1.sampler.custom_sampler.path=pkg://uni_agent.taskpilot.sampler \
    trainer.v1.sampler.custom_sampler.name=TaskPilotSampler \
    ++trainer.v1.sampler.sampler_kwargs.config_path="${TASKPILOT_CONFIG}" \
    ++trainer.v1.sampler.sampler_kwargs.iteration="${ITERATION}" \
    transfer_queue.enable=True \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
    ++actor_rollout_ref.rollout.custom.agent_framework.length_penalty.enable="${LENGTH_PENALTY_ENABLE}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.length_penalty.free_tokens="${LENGTH_PENALTY_FREE_TOKENS}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.length_penalty.alpha="${LENGTH_PENALTY_ALPHA}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console','tensorboard']" \
    trainer.nnodes="${TRAINER_NNODES}" \
    trainer.n_gpus_per_node="${TRAINER_GPUS_PER_NODE}" \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"
