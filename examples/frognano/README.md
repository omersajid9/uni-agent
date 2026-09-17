# FrogNano: single-node SWE RL for Qwen3-4B with TaskPilot band filtering

A 1x8 H100 recipe that trains a 4B agent on SWE tasks. It differs from the other
Uni-Agent recipes in one respect: **each update trains only on tasks that are
learnable for the current checkpoint.**

Rollout already produces 8 trajectories per task, so their mean pass rate *is* a
calibrated solve rate `p̂` for that task under the live policy. Groups outside the
iteration's target band are dropped and replaced with fresh prompts before the
update is formed, and the batch is filled with the groups closest to the target
rate. Calibration and training share the same rollouts, so the filter costs
generation for the discarded groups only -- no separate scoring pass.

| Piece | Where |
| --- | --- |
| Target-band schedule (data) | [`configs/taskpilot.yaml`](configs/taskpilot.yaml) |
| Per-step band filter | `uni_agent/taskpilot/sampler.py` (verl custom sampler) |
| Offline band split between iterations | `python -m uni_agent.taskpilot.calibrate` |
| Success-gated log-length penalty | `uni_agent/framework/length_penalty.py` (opt-in) |
| Agent + sandbox defaults | [`task_config_leaf.yaml`](task_config_leaf.yaml) |

## Target bands

```yaml
taskpilot:
  metric: acc
  target_band:
    iterations_1_to_4:
      target_resolve_rate: 0.5      # soft: keep 0 < p̂ < 1, prefer p̂ ≈ 0.5
    iteration_5:
      min_exclusive: 0.0            # hard: drop p̂ = 0 (too hard)
      max_inclusive: 0.5            # hard: drop p̂ > 0.5 (too easy for this stage)
```

`target_resolve_rate` ranks; `min_exclusive` / `max_inclusive` filter. With `n=8`
rollouts the default band is exactly "1/8 to 7/8 solved". Both compose, and the
band always reads the raw `acc` flag, so reward shaping can never move a group
across it.

Add iterations by adding keys (`iteration_6`, `iterations_6_to_10`, `default`);
nothing in the code enumerates them.

## Run one iteration

Each invocation is one ~200-update climb.

```bash
ITERATION=1 \
MODEL_PATH=~/models/Qwen3-4B \
TRAIN_FILE=~/data/frognano/iter1/accepted.parquet \
VAL_FILE=~/data/frognano/swe_bench_verified.parquet \
examples/frognano/train_frognano_qwen3_4b.sh
```

Node layout (`separate_async`): 2 GPUs train under FSDP2 with Ulysses SP 2, the
other 6 run single-GPU vLLM engines. 32 task groups x 8 trajectories = 256
trajectories per update, 64k context, GRPO with group-standardized advantages, no
KL, no entropy bonus.

Useful overrides: `TRAINER_GPUS_PER_NODE` / `ROLLOUT_GPUS_PER_NODE`,
`MAX_RESPONSE_LENGTH`, `CONCURRENCY`, `TOTAL_TRAINING_STEPS`, `TOOL_PARSER`
(must match the model's chat template).

The length penalty (`R(n) = 1 - alpha*ln(n/free_tokens)` past the free budget,
`0.5` for a correct-but-truncated trajectory) turns on automatically from
`ITERATION=3`; force it with `LENGTH_PENALTY_ENABLE`, and tune with
`LENGTH_PENALTY_FREE_TOKENS` / `LENGTH_PENALTY_ALPHA`.

## Between iterations

1. **Carry the weights, reset the optimizer.** Point `MODEL_PATH` at the previous
   iteration's exported HF checkpoint
   (`checkpoints/frognano/<exp>/global_step_N/huggingface`) and bump `ITERATION`.
2. **Re-calibrate the pool** against that checkpoint, since `p̂` is policy-relative
   and a pool saturates as the agent improves. Serve the checkpoint, then:

   ```bash
   BASE_URL=http://localhost:8000/v1 MODEL=Qwen3-4B \
       python examples/inference/parallel_infer_api.py \
       --data-path ~/data/frognano/pool.parquet \
       --task-config examples/frognano/task_config_leaf.yaml \
       --n 8 --result-path ~/data/frognano/iter2/rollouts.json

   python -m uni_agent.taskpilot.calibrate \
       --data-path ~/data/frognano/pool.parquet \
       --results ~/data/frognano/iter2/rollouts.json \
       --config examples/frognano/configs/taskpilot.yaml \
       --iteration 2 --out-dir ~/data/frognano/iter2
   ```

   This writes `accepted.parquet` (next `TRAIN_FILE`), `off_band.parquet` (the
   saturated / too-hard tasks to rewrite or drop), and `calibration.jsonl` with
   every task's `p̂` and disposition.

Offline calibration is optional -- the per-step filter already protects each
update -- but it keeps the pool from being mostly waste, which is what makes a
small train batch affordable on one node.

## Task pool

Any SWE dataset in Uni-Agent parquet format works, e.g.
`python -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ~/data/frognano`.
Synthetic candidates (generated issue + gold patch + hidden tests) belong in the
same format; validate them first with
`examples/inference/parallel_verify_swe.py`, which runs each row in oracle mode
and checks that the gold patch makes the F2P tests pass without breaking P2P.
Only rows that pass that gate are worth calibrating.
