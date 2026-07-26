#!/usr/bin/env bash
# =============================================================================
# Environment setup for: H200 NVL 141 GB, driver 570.211 / CUDA 12.8, 10 CPU / 32 GB host RAM, single GPU.
#
# Details about entire environment snapshot can be found in ./env_report.txt

# Run phases individually:  bash setup_uni_agent_env.sh phase1
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-uniagent}"
PYVER=3.12
REPO="${REPO:-$HOME/uni-agent}"
PINS="$REPO/pins.txt"

eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------------
phase0_checks() {
  echo "=== PHASE 0: verifying resources ==="

  echo "== driver / GPU =="
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
  echo "== host RAM =="
  free -g | head -2
  echo "== disk =="
  df -h "$HOME" | tail -1
  echo "== cuda toolkit (expect a STUB: few/no headers) =="
  ls /usr/local/cuda/include/cusparse.h 2>&1 || echo "  cusparse.h ABSENT (expected)"
}

# -----------------------------------------------------------------------------
phase1_base() {
  echo "=== PHASE 1: conda env + vLLM 0.19.1 + torch 2.10.0+cu128 ==="

  conda env remove -y -n "$ENV_NAME" 2>/dev/null || true
  conda create -y -n "$ENV_NAME" python=$PYVER
  conda activate "$ENV_NAME"

  # Pin versions of core libraries
  cat > "$PINS" <<'EOF'
torch==2.10.0
torchvision==0.25.0
torchaudio==2.10.0
vllm==0.19.1
EOF

  # vLLM pulls torch 2.10.0+cu128 and the whole nvidia-*-cu12 header/lib set.
  pip install --no-cache-dir "vllm==0.19.1"

  # define env variables for conda start up
  mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
  cat > "$CONDA_PREFIX/etc/conda/activate.d/zz-cuda.sh" <<'EOF'
NV="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
if [ -d "$NV" ]; then
  _inc="$(ls -d $NV/*/include 2>/dev/null | grep -v '/cu13/' | tr '\n' ':')"
  _lib="$(ls -d $NV/*/lib     2>/dev/null | grep -v '/cu13/' | tr '\n' ':')"
  export CPATH="${_inc}"
  export LIBRARY_PATH="${_lib}"
  export LD_LIBRARY_PATH="${_lib}/usr/local/cuda/lib64"
  unset _inc _lib
fi
export CUDA_HOME=/usr/local/cuda
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export MAX_JOBS=4
EOF

  conda deactivate; conda activate "$ENV_NAME"

  echo "--- gate 1: torch must be a CUDA 12 build and must reach the GPU ---"
  python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.version.cuda.startswith("12"), "CUDA 13 build cannot run on driver 570"
torch.zeros(1).cuda()
print("CUDA OK")
PY
}

# -----------------------------------------------------------------------------
phase2_verl() {
  echo "=== PHASE 2: verl + uni-agent ==="
  conda activate "$ENV_NAME"
  cd "$REPO"
  git submodule update --init --recursive

  pip install --no-cache-dir -c "$PINS" -e ./verl
  pip install --no-cache-dir -c "$PINS" -e .
  pip install --no-cache-dir -c "$PINS" \
      swe-rex loguru pydantic pydantic-settings aiohttp \
      modal swebench codetiming pylatexenc mathruler
  pip install --no-cache-dir -c "$PINS" --no-deps "TransferQueue==0.1.9"
  pip install --no-cache-dir -c "$PINS" --no-deps "ray[default]" "tensordict>=0.10.0"

  echo "--- gate 2: vLLM serves the model standalone (30 s, not a 5-min Ray launch) ---"
  echo "Run manually, Ctrl-C once you see 'Application startup complete':"
  echo "  python -m vllm.entrypoints.cli.main serve \$MODEL_PATH \\"
  echo "      --gpu-memory-utilization 0.20 --max-model-len 8192"
}

# -----------------------------------------------------------------------------
phase3_megatron() {
  echo "=== PHASE 3: Megatron stack (TE source build) ==="
  conda activate "$ENV_NAME"

  echo "--- headers must resolve BEFORE starting a 40-minute build ---"
  echo "$CPATH" | tr ':' '\n' | grep -E 'cusparse|cuda_runtime' || {
    echo "FATAL: cusparse/cuda_runtime not on CPATH"; return 1; }
  echo "$CPATH" | tr ':' '\n' | grep '/cu13/' && {
    echo "FATAL: cu13 headers on CPATH - remove those packages first"; return 1; }
  find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" -name cusparse.h | head -1

  pip install --no-cache-dir -c "$PINS" --no-deps "megatron-core==0.16.1"

  # No prebuilt wheels exist for TE 2.11.0 (checked: the GitHub release has zero
  # assets). transformer_engine_torch is an sdist and always compiles.
  pip install --no-cache-dir -c "$PINS" --no-build-isolation \
      "transformer_engine[core,pytorch]==2.11.0" 2>&1

  pip install --no-cache-dir -c "$PINS" --force-reinstall --no-deps \
      git+https://github.com/ISEEKYAN/mbridge

  pip install --no-cache-dir -c "$PINS" flash-linear-attention
  CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-cache-dir -c "$PINS" \
      --no-build-isolation --no-deps causal-conv1d

  pip install --no-cache-dir -c "$PINS" tilelang

  # install flash attention for cu12.8, torch2.10.0
  pip install --no-cache-dir -c "$PINS" --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
}

# -----------------------------------------------------------------------------
verify() {
  conda activate "$ENV_NAME"
  python - <<'PY'
import importlib, traceback
def chk(label, fn):
    try:
        print(f"  OK   {label}: {fn()}")
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}")

import torch
print("== core ==")
chk("torch", lambda: f"{torch.__version__} cuda={torch.version.cuda}")
chk("cuda device", lambda: (torch.zeros(1).cuda(), torch.cuda.get_device_name(0))[1])
chk("vllm", lambda: importlib.import_module("vllm").__version__)
chk("transformers", lambda: importlib.import_module("transformers").__version__)
chk("verl", lambda: importlib.import_module("verl").__file__)

print("== megatron stack (phase 3) ==")
chk("TE", lambda: importlib.import_module("transformer_engine.pytorch").__version__)
chk("TENorm", lambda: importlib.import_module("megatron.core.extensions.transformer_engine").TENorm.__name__)
chk("mcore", lambda: importlib.import_module("megatron.core").__version__)

print("== qwen3.5 only ==")
def bridge():
    from mbridge.core.auto_bridge import AutoBridge
    import mbridge.models
    from mbridge.models.qwen3_5 import Qwen3_5VlBridge
    return "qwen3_5 registered"
chk("mbridge qwen3_5", bridge)
chk("causal_conv1d", lambda: importlib.import_module("causal_conv1d").__name__)
chk("fla", lambda: importlib.import_module("fla.ops.gated_delta_rule").__name__)

print("== vllm arch support ==")
def archs():
    from vllm.model_executor.models.registry import ModelRegistry as R
    a = R.get_supported_archs()
    return [x for x in ("Qwen2ForCausalLM","Qwen3ForCausalLM", "Qwen3_5ForConditionalGeneration") if x in a]
chk("architectures", archs)
print("== TE support ==")
from importlib.metadata import version
chk("TE", lambda: version("transformer-engine"))
chk("tilelang", lambda: importlib.import_module("tilelang").__version__)
chk("flash_attn", lambda: importlib.import_module("flash_attn").__version__)
chk("transfer_queue", lambda: importlib.import_module("transfer_queue").__name__)
PY
}

case "${1:-all}" in
  phase0)  phase0_checks ;;
  phase1)  phase1_base ;;
  phase2)  phase2_verl ;;
  phase3)  phase3_megatron ;;
  verify)  verify ;;
  all)     phase0_checks; phase1_base; phase2_verl; phase3_megatron; verify ;;
  *) echo "usage: $0 {phase0|phase1|phase2|phase3|verify|all}"; exit 1 ;;
esac