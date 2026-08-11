# ssail environment & training scripts

Personal setup/launch scripts for `uni-agent`. Every script here is routed
by `DEVICE` (`gpu` or `neuron`, default `neuron`).

## 1. Configure secrets

```bash
cp .env.example .env    # already gitignored — never commit real tokens
```

Fill in `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `.env`.

## 2. Set up the environment

```bash
bash setup_env.sh              # neuron (default) -> env/setup_env_trn.sh
bash setup_env.sh gpu          # GPU              -> env/setup_env_gpu.sh (all phases)
bash setup_env.sh gpu phase1   # run a single GPU phase; extra args pass straight through
```

## 3. Activate the environment

`setup_env.sh` only activates the environment inside its own subshell, so
activate it again in your interactive shell before running anything else:

```bash
# GPU
conda activate uniagent

# Neuron
conda activate mini_verl_neuron
```

## 4. Prepare data

Neuron: handled automatically as the last step of `setup_env.sh` (skip with
`SKIP_DATA_PREP=1` and run it later). GPU: run it yourself first:

```bash
bash scripts/setup_swe_data.sh
```

Both write parquet files under `$HOME/uni-agent-data/data/uni_agent/` by
default (override with `DATA_REPO=...`), which is also where the `gpu`/`neuron`
launchers look for them by default (via `DATA_DIR`).

GPU also needs a local model snapshot:

```bash
# log into hf
hf auth login

# download model locally
hf download Qwen/Qwen3-4B --local-dir "$HOME/uni-agent-data/models/Qwen3-4B"
```

## 5. Bring up Ray

Both devices submit via `ray job submit`, so a reachable cluster must already
be running. Single-node dev example — adjust `--num-cpus`/`--num-gpus` to your box:

On 1 H200 instance:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=9 --num-gpus=1 --include-dashboard=true
```

On trn2.3xlarge instance:
```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=10 --include-dashboard=true
```

## 6. Run training

```bash
bash start_train.sh        # neuron (default)
SINGLE_NODE=true bash start_train.sh gpu    # GPU
```

Each launcher (`gpu/train_qwen3p5_dense.sh`, `neuron/train_qwen3p5_dense.sh`)
generates its Ray `runtime_env.yaml` into a tempfile on every run — it embeds
your Modal tokens and (for GPU) this machine's conda nvidia lib paths, so it's
never checked into git. Override `RUNTIME_ENV=/path/to/your.yaml` to supply
your own instead.

## 7. View Ray Dashboard using Cloudflare

Install Cloudflare:
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

Open reverse proxy:
```
cloudflared tunnel --url http://localhost:8265
```
