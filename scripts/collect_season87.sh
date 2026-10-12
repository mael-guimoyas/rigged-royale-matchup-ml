#!/usr/bin/env bash
# Relay for a Netcup unit installed before season 88: it still runs this path.
# Re-run scripts/install_netcup_collector.sh to install the season 88 unit (its
# environment differs: COLLECTOR_DEFICIT_BIAS=0 instead of 0.5).
set -euo pipefail
echo "collect_season87.sh is deprecated: re-run scripts/install_netcup_collector.sh" >&2
exec /bin/bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/collect_season88.sh" "$@"
