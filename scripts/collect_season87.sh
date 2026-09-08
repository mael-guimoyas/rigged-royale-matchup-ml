#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collector_id="${COLLECTOR_ID:-netcup}"
cycle_hours="${COLLECTOR_CYCLE_HOURS:-6}"
requests_per_second="${COLLECTOR_RPS:-30}"
token_mode="${CR_API_TOKEN_MODE:-1}"
storage_bucket="${STORAGE_BUCKET:-training-battles}"
storage_prefix="${STORAGE_PREFIX:-season-87}"

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
  --stats-interval 60
