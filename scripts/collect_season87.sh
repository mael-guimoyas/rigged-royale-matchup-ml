#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collector_id="${COLLECTOR_ID:-netcup}"
cycle_hours="${COLLECTOR_CYCLE_HOURS:-6}"
requests_per_second="${COLLECTOR_RPS:-30}"
token_mode="${CR_API_TOKEN_MODE:-1}"
storage_bucket="${STORAGE_BUCKET:-training-battles}"
storage_prefix="${STORAGE_PREFIX:-season-87}"
# Catch-up steering for the sub-14000 ladder. Proportional collection mirrors
# the live population, and the live population is top-heavy: at 43.1M battles
# season 87 sat at 51.2% in ladder:14000-14999 while every band below 12000 held
# under 1.5%, and 5000-5999 under 0.5%. With this probability the frontier draws
# the next player from the most starved band instead, so the thin bands fill
# while the saturated one idles.
#
# The cost is representativeness inside those bands: steering aims at the
# players passing through a band rather than the ones living in it, which is
# what skewed the fetched side to 63.5% wins in ladder:5000-6999 against 37.2%
# in 14000+ the last time it was on. So this is a top-up setting, not a
# permanent one -- set COLLECTOR_DEFICIT_BIAS=0 to return to representative
# collection once the low bands are deep enough.
#
# 0.5, not higher: measured on the 43.1M battles collected proportionally, the
# same deck already scores 3.2 points lower on average (10.7 at worst) when it
# is the fetched side than when it is the opponent. That residual is what the
# steering will widen, and the 9-24 point version of it is what made the last
# steered corpus unusable. Raise this only against the win rates the collector
# now reports per band at the end of every cycle.
deficit_bias="${COLLECTOR_DEFICIT_BIAS:-0.5}"
# A starved band is only served once its queue holds this many candidates, so a
# steered draw stays a random sample of the band instead of emptying it.
min_queue_depth="${COLLECTOR_MIN_QUEUE_DEPTH:-500}"

exec "${repo_root}/.venv-collector/bin/rigged-collector" \
  --config "${repo_root}/config/default.yaml" \
  --tags-file "${repo_root}/seeds_balanced.txt" \
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
  --stats-interval 60
