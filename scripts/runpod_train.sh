#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mael-guimoyas/rigged-royale-matchup-ml.git}"
WORKDIR="${WORKDIR:-/workspace/rigged-royale-matchup-ml}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw}"
PREPARED_DIR="${PREPARED_DIR:-$DATA_ROOT/prepared}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/artifacts}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/runpod.yaml}"
VENV_DIR="${VENV_DIR:-/workspace/rrm-venv}"
export RAW_DIR PREPARED_DIR ARTIFACT_DIR CONFIG_PATH

log_step() {
  echo "[$(date -Is)] $*"
}

if [[ ! -f "pyproject.toml" || ! -d "src/rigged_matchup_ml" ]]; then
  if [[ -d "$WORKDIR/.git" ]]; then
    cd "$WORKDIR"
    git pull --ff-only
  else
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
  fi
fi

mkdir -p "$RAW_DIR" "$PREPARED_DIR" "$ARTIFACT_DIR"

log_step "Installing Python package"
python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
# uv resolves/installs ~10-30x faster than pip. Fall back to pip if uv fails so
# the run never gets stuck on the install step. Install runtime deps only (no
# [dev]); torch is reused from the base image via --system-site-packages.
if python -m pip install --upgrade uv && python -m uv pip install -e .; then
  log_step "Installed with uv"
else
  log_step "uv unavailable or failed; falling back to pip"
  python -m pip install -e .
fi

log_step "Writing RunPod config"
python - <<'PY'
import os
from pathlib import Path

import yaml


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


with open("config/default.yaml", "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["data"]["raw_dir"] = os.environ["RAW_DIR"]
cfg["data"]["prepared_dir"] = os.environ["PREPARED_DIR"]
cfg["training"]["artifact_dir"] = os.environ["ARTIFACT_DIR"]
cfg["training"]["device"] = "auto"
cfg["training"]["batch_size"] = env_int("RUNPOD_BATCH_SIZE", 4096)
cfg["training"]["evaluation_batch_size"] = env_int("RUNPOD_EVAL_BATCH_SIZE", 8192)
cfg["training"]["gradient_accumulation_steps"] = env_int("RUNPOD_GRAD_ACCUM", 1)
cfg["training"]["num_workers"] = env_int("RUNPOD_NUM_WORKERS", 4)
cfg["training"]["epochs"] = env_int("RUNPOD_EPOCHS", int(cfg["training"]["epochs"]))

config_path = Path(os.environ["CONFIG_PATH"])
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(f"wrote {config_path}")
PY

log_step "Checking CUDA"
python - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Pick a GPU Pod/template before training.")
PY

if [[ ! -f "$PREPARED_DIR/manifest.json" ]]; then
  if [[ -n "${SUPABASE_URL:-}" && -n "${SUPABASE_SECRET_KEY:-}" ]]; then
    log_step "Pulling raw Parquet shards from Supabase Storage"
    rigged-matchup pull-storage --config "$CONFIG_PATH" \
      --bucket "${TRAINING_BUCKET:-training-battles}" \
      --prefix "${TRAINING_PREFIX:-battles}" \
      --workers "${STORAGE_DOWNLOAD_WORKERS:-16}"
    log_step "Preparing chronological train/validation/test splits"
    rigged-matchup prepare --config "$CONFIG_PATH" --overwrite
  else
    echo "Missing $PREPARED_DIR/manifest.json."
    echo "Upload prepared data there, or set SUPABASE_URL and SUPABASE_SECRET_KEY to pull raw shards."
    exit 1
  fi
fi

if [[ ! -f "$PREPARED_DIR/card2vec.npy" ]]; then
  log_step "Pretraining card embeddings"
  rigged-matchup pretrain-cards --config "$CONFIG_PATH"
else
  log_step "Skipping card2vec pretrain; $PREPARED_DIR/card2vec.npy already exists"
fi

if [[ "${RUN_ATTACH_PRIOR:-0}" == "1" ]]; then
  log_step "Attaching empirical prior"
  rigged-matchup attach-prior --config "$CONFIG_PATH"
fi

log_step "Starting model training"
rigged-matchup train --config "$CONFIG_PATH" 2>&1 | tee "$ARTIFACT_DIR/train.log"
log_step "Evaluating checkpoint"
rigged-matchup evaluate "$ARTIFACT_DIR/matchup-model.pt" --config "$CONFIG_PATH" \
  2>&1 | tee "$ARTIFACT_DIR/evaluate.log"

if [[ "${RUN_BENCHMARK:-1}" == "1" ]]; then
  log_step "Running benchmark"
  rigged-matchup benchmark "$ARTIFACT_DIR/matchup-model.pt" --config "$CONFIG_PATH" \
    --split test --min-support "${BENCHMARK_MIN_SUPPORT:-100}" \
    2>&1 | tee "$ARTIFACT_DIR/benchmark.log"
fi

echo "Done. Artifacts are in $ARTIFACT_DIR"
