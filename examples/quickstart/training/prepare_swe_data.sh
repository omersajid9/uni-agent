#!/usr/bin/env bash
# Preprocess SWE-reBench (train) and SWE-Bench Verified (val) into the parquet the mini-verl
# SWE launcher reads. Each row carries the chat prompt + a per-sample uni_agent Task Config
# (name, sandbox image, metadata); SweDataset forwards them to the agentic rollout.
#
# Run once before train_swe_miniverl.sh. Defaults produce a tiny smoke set; drop the caps for a
# full run.
#
#   bash examples/quickstart/training/prepare_swe_data.sh
#   SAVE_DIR=~/data/uni_agent MAX_INSTANCES=500 bash examples/quickstart/training/prepare_swe_data.sh
#
# Writes:
#   <SAVE_DIR>/swe_rebench_filtered.parquet   (train)
#   <SAVE_DIR>/swe_bench_verified.parquet     (val)
set -euo pipefail

SAVE_DIR=${SAVE_DIR:-"${HOME}/data/uni_agent"}
# Cap for the smoke run. Set MAX_INSTANCES= (empty) or a large number for the full dataset.
MAX_INSTANCES=${MAX_INSTANCES:-32}

mkdir -p "${SAVE_DIR}"

EXTRA_ARGS=()
if [[ -n "${MAX_INSTANCES}" ]]; then
    EXTRA_ARGS+=(--max-instances "${MAX_INSTANCES}")
fi

echo "==> SWE-reBench (train) -> ${SAVE_DIR}/swe_rebench_filtered.parquet"
python -m uni_agent.tasks.swe_rebench.preprocess \
    --local-save-dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "==> SWE-Bench Verified (val) -> ${SAVE_DIR}/swe_bench_verified.parquet"
python -m uni_agent.tasks.swe_bench.preprocess \
    --local-save-dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "done. train_swe_miniverl.sh defaults to DATA_DIR=${SAVE_DIR}"
