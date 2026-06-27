#!/usr/bin/env bash
# Ceiling analysis on a RunPod: estimate the model's theoretical ceiling and
# print where it is weak + how to get closer. Run this AFTER scripts/runpod_train.sh
# (it reuses the same venv, config, prepared data and trained checkpoint).
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/rigged-royale-matchup-ml}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
PREPARED_DIR="${PREPARED_DIR:-$DATA_ROOT/prepared}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/artifacts}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/runpod.yaml}"
VENV_DIR="${VENV_DIR:-/workspace/rrm-venv}"
CHECKPOINT="${CHECKPOINT:-$ARTIFACT_DIR/matchup-model.pt}"
CEILING_SPLIT="${CEILING_SPLIT:-test}"
CEILING_MIN_SUPPORT="${CEILING_MIN_SUPPORT:-100}"

log_step() {
  echo "[$(date -Is)] $*"
}

# Land in the repo so `config/default.yaml` and the package import resolve.
if [[ ! -f "pyproject.toml" || ! -d "src/rigged_matchup_ml" ]]; then
  if [[ -d "$WORKDIR/.git" ]]; then
    cd "$WORKDIR"
  else
    echo "Repo not found at $WORKDIR. Run scripts/runpod_train.sh first or set WORKDIR."
    exit 1
  fi
fi

# Reuse the training venv if present; otherwise install the runtime package.
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
else
  log_step "venv not found at $VENV_DIR; installing the package"
  python -m venv --system-site-packages "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -e .
fi

# Fall back to the repo default config if the RunPod one was not written.
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="config/default.yaml"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT. Train first (scripts/runpod_train.sh) or set CHECKPOINT."
  exit 1
fi
if [[ ! -d "$PREPARED_DIR/$CEILING_SPLIT" ]]; then
  echo "Prepared split not found: $PREPARED_DIR/$CEILING_SPLIT. Run `rigged-matchup prepare`."
  exit 1
fi

log_step "Analysing theoretical ceiling (split=$CEILING_SPLIT, min_support=$CEILING_MIN_SUPPORT)"
rigged-matchup ceiling "$CHECKPOINT" --config "$CONFIG_PATH" \
  --split "$CEILING_SPLIT" --min-support "$CEILING_MIN_SUPPORT" \
  2>&1 | tee "$ARTIFACT_DIR/ceiling.log"

echo "JSON report: $ARTIFACT_DIR/ceiling-$CEILING_SPLIT-report.json"
echo "Log:         $ARTIFACT_DIR/ceiling.log"
