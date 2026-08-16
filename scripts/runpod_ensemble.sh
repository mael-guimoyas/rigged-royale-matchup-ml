#!/usr/bin/env bash
# Train several checkpoints that differ only by their seed.
#
# Why an ensemble at all: the optimizer picks the highest-scoring deck out of
# thousands of candidates that have never been played, so it lands on whichever
# deck the model overrates. Independently trained members disagree most exactly
# where the data does not constrain them, so their spread is a per-deck
# uncertainty the single checkpoint cannot produce, and selecting with one
# member while displaying another removes the selection bias exactly.
#
# The shared work — Supabase download, chronological splits, card2vec — happens
# once, inside the first member's run. Every later member reuses it, which is
# both a large time saving and a correctness requirement: members trained on
# different splits would not be comparable.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
ENSEMBLE_DIR="${ENSEMBLE_DIR:-/workspace/ensemble}"
SEEDS="${SEEDS:-101 202 303 404}"
# The extra members exist to be averaged and to measure spread, so the slow
# per-member benchmark is off by default. The evaluation stays on: it is what
# proves each member is individually sound rather than a failed run.
RUN_BENCHMARK="${RUN_BENCHMARK:-0}"
export DATA_ROOT RUN_BENCHMARK

log_step() {
  echo "[$(date -Is)] $*"
}

mkdir -p "$ENSEMBLE_DIR"

for seed in $SEEDS; do
  target="$ENSEMBLE_DIR/seed-$seed"
  if [[ -s "$target/matchup-model.pt" ]]; then
    log_step "seed $seed already trained at $target, skipping"
    continue
  fi
  log_step "=== training ensemble member, seed $seed ==="
  mkdir -p "$target"
  RUNPOD_SEED="$seed" \
  ARTIFACT_DIR="$target" \
  CONFIG_PATH="/workspace/runpod-seed-$seed.yaml" \
    bash scripts/runpod_train.sh
  log_step "seed $seed done -> $target/matchup-model.pt"
done

log_step "=== ensemble complete ==="
ls -lh "$ENSEMBLE_DIR"/seed-*/matchup-model.pt

# Two members that produced byte-identical weights would mean the seed override
# never reached the trainer, and the whole run would be worthless. Checking is
# cheaper than discovering it after the download.
log_step "checkpoint fingerprints (these must all differ)"
sha256sum "$ENSEMBLE_DIR"/seed-*/matchup-model.pt
