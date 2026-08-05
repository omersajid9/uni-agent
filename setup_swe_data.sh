#!/usr/bin/env bash
# Preprocess SWE-reBench (train) and SWE-Bench Verified (val) into the parquet files that the
# mini-verl SWE launchers (examples/quickstart/training/mini-verl/train_*.sh) read.
#
#
#   bash setup_swe_data.sh
#   DATA_REPO=~/other-data MAX_INSTANCES=500 bash setup_swe_data.sh   # full dataset
#
# Writes:
#   <DATA_REPO>/data/uni_agent/swe_rebench_filtered.parquet   (train)
#   <DATA_REPO>/data/uni_agent/swe_bench_verified.parquet     (val)
set -euo pipefail

DATA_REPO="${DATA_REPO:-$HOME/uni-agent-data}"
SAVE_DIR="${SAVE_DIR:-$DATA_REPO/data/uni_agent}"
# Cap for a smoke run. Set MAX_INSTANCES= (empty) or a large number for the full dataset.
MAX_INSTANCES="${MAX_INSTANCES:-32}"

python -c "import uni_agent" >/dev/null 2>&1 || {
    echo "ERROR: uni_agent is not importable in this environment." >&2
    echo "  Install it first, e.g. bash setup_env_trn.sh (or setup_env_gpu.sh)." >&2
    exit 1
}

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

echo "done. DATA_DIR=${DATA_REPO} for the mini-verl launchers (train/test files under data/uni_agent/)."
