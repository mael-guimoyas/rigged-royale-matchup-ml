#!/usr/bin/env bash
# Relay for a Netcup unit installed during season 88: it still runs this path.
# Without the relay it kept uploading season 89 battles under the season-88
# Storage prefix (shards 000939-001901, moved to season-89 on 2026-11-13).
# Re-run scripts/install_netcup_collector.sh to install the season 89 unit.
set -euo pipefail
echo "collect_season88.sh is deprecated: re-run scripts/install_netcup_collector.sh" >&2
exec /bin/bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/collect_season89.sh" "$@"
