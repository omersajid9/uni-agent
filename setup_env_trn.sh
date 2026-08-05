#!/usr/bin/env bash
# Bootstrap the AWS Neuron (Trainium) environment for uni-agent + mini-verl + verl.
#
# Layout assumption (override with env vars if yours differs):
#   REPO       ($HOME/uni-agent)       - this code repo; `verl` and `mini-verl` are its git
#                                        submodules (REPO/verl, REPO/mini-verl).
#   DATA_REPO  ($HOME/uni-agent-data)  - datasets/checkpoints/logs, kept OUT of the code repo.
#
# Usage:
#   bash setup_env_trn.sh
#   SKIP_DATA_PREP=1 bash setup_env_trn.sh          # skip dataset prep, wire it up later
#   VENV_ACTIVATE=/no/such/path bash setup_env_trn.sh  # force the uv-venv fallback path
set -euo pipefail

REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_REPO="${DATA_REPO:-$HOME/uni-agent-data}"
MINI_VERL_DIR="${MINI_VERL_DIR:-$REPO/mini-verl}"
VERL_DIR="${VERL_DIR:-$REPO/verl}"

VENV_ACTIVATE="${VENV_ACTIVATE:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate}"
# Fallback venv used only when $VENV_ACTIVATE is not found.
UV_VENV_DIR="${UV_VENV_DIR:-$REPO/.venv-neuron}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

NEURONX_COMPILED_WHEEL="${NEURONX_COMPILED_WHEEL:-$REPO/torch_neuronx-2.11.3.0.19138+4dee388.dev-cp312-cp312-linux_x86_64.whl}"

# Pinned versions from mini-verl/mini_verl_neuron_guideline.md — keep these two in sync.
NEURON_DKMS_VERSION="${NEURON_DKMS_VERSION:-2.28.0.0}"
NEURON_COLLECTIVES_VERSION="${NEURON_COLLECTIVES_VERSION:-2.32.28.0-452cba8de}"
NEURON_RUNTIME_LIB_VERSION="${NEURON_RUNTIME_LIB_VERSION:-2.32.31.0-0234f5ed2}"
NEURONX_CC_VERSION="${NEURONX_CC_VERSION:-2.25.3371.0+f524f7f8}"

SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

# Installs into whichever environment is currently active: `uv pip` in the uv-venv fallback
# (a bare `uv venv` has no pip binary by design), `python -m pip` in the prebuilt AWS venv.
pip_install() {
    if [[ "${USE_UV:-0}" == "1" ]]; then
        uv pip install "$@"
    else
        python -m pip install "$@"
    fi
}

pip_uninstall() {
    if [[ "${USE_UV:-0}" == "1" ]]; then
        uv pip uninstall "$@"
    else
        python -m pip uninstall -y "$@"
    fi
}

# -----------------------------------------------------------------------------------------
# A) git submodule sync + recursive init
# -----------------------------------------------------------------------------------------
[[ -f "$REPO/pyproject.toml" ]] || die "uni-agent repository not found at REPO=$REPO"

log "Syncing and initializing git submodules (verl, mini-verl) under $REPO"
git -C "$REPO" submodule sync --recursive
git -C "$REPO" submodule update --init --recursive

[[ -f "$MINI_VERL_DIR/pyproject.toml" ]] || die "mini-verl submodule not found: $MINI_VERL_DIR"
[[ -f "$VERL_DIR/setup.py" || -f "$VERL_DIR/pyproject.toml" ]] || die "verl submodule not found: $VERL_DIR"

# -----------------------------------------------------------------------------------------
# B) Environment: reuse the prebuilt AWS Neuron venv, or bootstrap one with uv
# -----------------------------------------------------------------------------------------
if [[ -f "$VENV_ACTIVATE" ]]; then
    log "Found prebuilt AWS Neuron venv, activating: $VENV_ACTIVATE"
    USE_UV=0
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    log "No prebuilt AWS Neuron venv at $VENV_ACTIVATE — bootstrapping a fresh uv venv"
    USE_UV=1

    if ! command -v uv >/dev/null 2>&1; then
        log "uv not found, installing (https://astral.sh/uv)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || die "uv installation failed; install manually: https://docs.astral.sh/uv/"

    if [[ ! -f "$UV_VENV_DIR/bin/activate" ]]; then
        log "Creating uv venv (python $PYTHON_VERSION) at $UV_VENV_DIR"
        uv venv --python "$PYTHON_VERSION" "$UV_VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$UV_VENV_DIR/bin/activate"

    # --- mini_verl_neuron_guideline.md Step 2: Neuron driver / collectives / runtime lib ---
    installed_dkms="$(dpkg-query -W -f='${Version}' aws-neuronx-dkms 2>/dev/null || true)"
    if [[ "$installed_dkms" == "$NEURON_DKMS_VERSION" ]]; then
        log "Neuron driver/collectives/runtime-lib already at pinned versions, skipping apt install"
    else
        command -v apt-get >/dev/null 2>&1 || die \
            "apt-get not found; install the Neuron driver stack manually (see mini-verl/mini_verl_neuron_guideline.md Step 2)"
        log "Installing Neuron driver/collectives/runtime-lib ${NEURON_DKMS_VERSION} (requires sudo)"
        # shellcheck disable=SC1091
        . /etc/os-release
        sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
        wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
        sudo apt-get update -y
        sudo apt-get install -y \
            aws-neuronx-dkms="${NEURON_DKMS_VERSION}" \
            aws-neuronx-collectives="${NEURON_COLLECTIVES_VERSION}" \
            aws-neuronx-runtime-lib="${NEURON_RUNTIME_LIB_VERSION}"
    fi

    # --- Step 3: neuronx-cc compiler ---
    if neuronx-cc --version 2>/dev/null | grep -qF "$NEURONX_CC_VERSION"; then
        log "neuronx-cc already at ${NEURONX_CC_VERSION}, skipping"
    else
        log "Installing neuronx-cc==${NEURONX_CC_VERSION}"
        pip_install --extra-index-url https://pip.repos.neuron.amazonaws.com "neuronx-cc==${NEURONX_CC_VERSION}"
    fi

fi

# --- Step 4: torch_neuronx from the local compiled wheel ---
[[ -f "$NEURONX_COMPILED_WHEEL" ]] || die \
    "torch_neuronx wheel not found. Set NEURONX_COMPILED_WHEEL to the compiled wheel path" \
    "(currently: $NEURONX_COMPILED_WHEEL). See mini-verl/mini_verl_neuron_guideline.md Step 4."
log "Installing torch_neuronx from $NEURONX_COMPILED_WHEEL"
pip_install "$NEURONX_COMPILED_WHEEL"


TORCH_VERSION_BEFORE="$(python -c 'import torch; print(torch.__version__)')" \
    || die "The selected environment does not contain PyTorch"
log "Using torch $TORCH_VERSION_BEFORE from $(command -v python)"

# -----------------------------------------------------------------------------------------
# Runtime dependency stack (mini_verl_neuron_guideline.md Step 5, plus uni-agent/mini-verl
# extras). Pinned so `verl`/`mini-verl` are installed with --no-deps below and never pull in
# CUDA/vLLM packages or clobber the Neuron-provided torch. `numpy>=2.0.0` is listed first and
# in the SAME command as everything else so pip's resolver honors it even though verl==0.8.0
# (if ever pulled in transitively) would otherwise pin numpy<2.0 — see guideline Step 5.1.
# -----------------------------------------------------------------------------------------
log "Installing pinned runtime dependencies"
pip_install \
    "numpy>=2.0.0" \
    "ray[default]==2.55.0" \
    "transformers==5.12.1" \
    "tensordict==0.10.0" \
    "pandas==3.0.3" \
    "pyarrow==24.0.0" \
    "omegaconf==2.3.0" \
    "accelerate==1.14.0" \
    codetiming datasets dill hydra-core packaging peft pybind11 pylatexenc \
    torchdata wandb fastapi uvicorn \
    swe-rex loguru "pydantic<3" pydantic-settings aiohttp \
    modal swebench mathruler \
    tensorboard msgspec cachetools
pip_install --no-deps "TransferQueue==0.1.8"

log "Uninstalling torchvision (causes conflict with torch)"
pip_uninstall torchvision || true

# -----------------------------------------------------------------------------------------
# C) verl — always from the local submodule, never from PyPI, so repo-local patches win.
# -----------------------------------------------------------------------------------------
log "Installing verl (editable, --no-deps) from $VERL_DIR"
pip_install --no-deps -e "$VERL_DIR"

# -----------------------------------------------------------------------------------------
# D) mini-verl, then uni-agent — both editable + --no-deps (all deps handled above without
#    touching the Neuron stack).
# -----------------------------------------------------------------------------------------
log "Installing mini-verl (editable, --no-deps) from $MINI_VERL_DIR"
pip_install --no-deps -e "$MINI_VERL_DIR"

log "Installing uni-agent (editable, --no-deps) from $REPO"
pip_install --no-deps -e "$REPO"

# -----------------------------------------------------------------------------------------
# Verify the full module chain imports and that torch was not silently replaced.
# -----------------------------------------------------------------------------------------
log "Verifying installation"
python - <<PY
import importlib

import torch

expected_torch = "$TORCH_VERSION_BEFORE"
assert torch.__version__ == expected_torch, (
    f"torch changed during installation: {expected_torch} -> {torch.__version__}"
)

for module in (
    "ray",
    "transformers",
    "tensordict",
    "verl",
    "mini_verl.ray_trainer",
    "mini_verl.workers.actor",
    "mini_verl.workers.rollout",
    "uni_agent",
):
    importlib.import_module(module)

try:
    import torch_neuronx
    print("torch_neuronx:", torch_neuronx.__file__)
except ImportError:
    print("torch_neuronx: not installed (CPU-only environment)")

import mini_verl
import uni_agent
import verl

print("verl:", getattr(verl, "__version__", "unknown"), verl.__file__)
print("mini-verl:", mini_verl.__file__)
print("uni-agent:", uni_agent.__file__)
print("Neuron environment setup complete")
PY

# -----------------------------------------------------------------------------------------
# E) Data repo: hand off to setup_swe_data.sh (trn-independent — no Neuron packages needed).
# -----------------------------------------------------------------------------------------
if [[ "$SKIP_DATA_PREP" == "1" ]]; then
    log "SKIP_DATA_PREP=1, skipping dataset preparation (run setup_swe_data.sh manually later)"
else
    log "Preparing SWE datasets in the data repo ($DATA_REPO)"
    DATA_REPO="$DATA_REPO" bash "$REPO/setup_swe_data.sh"
fi

log "Done. REPO=$REPO  DATA_REPO=$DATA_REPO"
