#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collector_id="${COLLECTOR_ID:-netcup}"
cycle_hours="${COLLECTOR_CYCLE_HOURS:-6}"
requests_per_second="${COLLECTOR_RPS:-30}"
token_mode="${CR_API_TOKEN_MODE:-1}"
storage_bucket="${STORAGE_BUCKET:-training-battles}"
storage_prefix="${STORAGE_PREFIX:-season-88}"
# Season 87 landed 51.2% of its battles in ladder:14000-14999 and under 1.5% in
# every band below 12000. The cause was the frontier, not the seeds: its global
# 100,000-tag queue filled within the first hour, mostly with 14000 players
# (the most active ones), and from then on refused every candidate of every
# band, so the proportional draw kept serving the frozen top-heavy mix.
#
# The queue is now capped per band (default: the global cap split evenly over
# the tracked bands), so a saturated band stops growing while the thin ones keep
# filling, and the draw flattens as bands reach the cap -- with a uniform draw
# over a deep queue inside each band. Seeds come from
# scripts/build_seeds_from_supabase.py, an equal quota per 1,000-trophy band.
#
# That makes the deficit steering unnecessary to start a season: it aimed at the
# players passing through a starved band rather than the ones living in it (the
# fetched side won 63.5% in ladder:5000-6999 against 37.2% in 14000+ the last
# time it ran hard). It defaults to off; COLLECTOR_DEFICIT_BIAS=0.5 brings back
# the season 87 top-up setting if a band still lags after a few cycles.
deficit_bias="${COLLECTOR_DEFICIT_BIAS:-0}"
# A starved band is only served once its queue holds this many candidates, so a
# steered draw stays a random sample of the band instead of emptying it.
min_queue_depth="${COLLECTOR_MIN_QUEUE_DEPTH:-500}"
# Empty = the collector default (global queue / tracked bands). 0 disables.
queue_per_band="${COLLECTOR_QUEUE_PER_BAND:-}"
seeds_file="${COLLECTOR_SEEDS_FILE:-${repo_root}/seeds_season88.txt}"
per_band_args=()
if [[ -n "${queue_per_band}" ]]; then
  per_band_args=(--max-queue-per-band "${queue_per_band}")
fi

exec "${repo_root}/.venv-collector/bin/rigged-collector" \
  --config "${repo_root}/config/default.yaml" \
  --tags-file "${seeds_file}" \
  --max-duration-hours "${cycle_hours}" \
  --requests-per-second "${requests_per_second}" \
  --workers 0 \
  --api-token-mode "${token_mode}" \
  --upload \
  --bucket "${storage_bucket}" \
  --prefix "${storage_prefix}" \
  --collector-id "${collector_id}" \
  --deficit-bias "${deficit_bias}" \
  --min-queue-depth "${min_queue_depth}" \
  ${per_band_args[@]+"${per_band_args[@]}"} \
  --stats-interval 60
