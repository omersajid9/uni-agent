#!/usr/bin/env bash
# Install mini-verl and uni-agent into the AWS Neuron PyTorch 2.9 environment.
set -euo pipefail

VENV_ACTIVATE="${VENV_ACTIVATE:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate}"
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
MINI_VERL_DIR="${MINI_VERL_DIR:-$REPO/mini-verl}"
VERL_VERSION="${VERL_VERSION:-0.9.0}"

# Replace this with the local compiled torch_neuronx wheel.
NEURONX_COMPILED_WHEEL="${NEURONX_COMPILED_WHEEL:-/path/to/torch_neuronx_compiled.whl}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f "$VENV_ACTIVATE" ]] || die "Neuron environment not found: $VENV_ACTIVATE"
[[ -f "$REPO/pyproject.toml" ]] || die "uni-agent repository not found: $REPO"
[[ -f "$MINI_VERL_DIR/pyproject.toml" ]] || die "mini-verl repository not found: $MINI_VERL_DIR"
[[ -f "$NEURONX_COMPILED_WHEEL" ]] || die \
    "Set NEURONX_COMPILED_WHEEL to the compiled wheel path (currently: $NEURONX_COMPILED_WHEEL)"

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# mini-verl currently declares Python >=3.12. Fail before changing the environment
# if the AWS image does not satisfy that requirement.
python - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(
        f"mini-verl requires Python >=3.12; the selected environment has {sys.version.split()[0]}"
    )
print("Using Python", sys.version.split()[0], "from", sys.executable)
PY

TORCH_VERSION_BEFORE="$(
    python -c 'import torch; print(torch.__version__)'
)" || die "The selected environment does not contain PyTorch"
echo "Preserving prebuilt torch $TORCH_VERSION_BEFORE"

# Never let the local Neuron wheel replace the AWS environment's PyTorch.
python -m pip install --no-deps "$NEURONX_COMPILED_WHEEL"

# Runtime versions validated by mini_verl_neuron_guideline.md. These are
# installed explicitly because mini-verl and verl are installed with --no-deps
# below to prevent pip from pulling CUDA/vLLM packages or replacing torch.
python -m pip install \
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
    modal swebench mathruler

# Use the released verl package. Do not initialize or install ./verl.
python -m pip install --no-deps "verl==$VERL_VERSION"
python -m pip install --no-deps "TransferQueue==0.1.8"

# Install only each repository's own Python package. --no-deps is intentional:
# all dependencies are handled above without touching the Neuron stack.
python -m pip install --no-deps -e "$MINI_VERL_DIR"
python -m pip install --no-deps -e "$REPO"

# export MINI_VERL_DEVICE=cpu
python - <<PY
import importlib
import torch
import torch_neuronx
import verl
import mini_verl
import uni_agent

expected_torch = "$TORCH_VERSION_BEFORE"
assert torch.__version__ == expected_torch, (
    f"torch changed during installation: {expected_torch} -> {torch.__version__}"
)

for module in (
    "ray",
    "transformers",
    "tensordict",
    "mini_verl.ray_trainer",
    "mini_verl.workers.actor",
    "mini_verl.workers.rollout",
):
    importlib.import_module(module)

print("torch_neuronx:", torch_neuronx.__file__)
print("verl:", getattr(verl, "__version__", "unknown"), verl.__file__)
print("mini-verl:", mini_verl.__file__)
print("uni-agent:", uni_agent.__file__)
print("Neuron environment setup complete")
PY
