#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_template="${repo_root}/deploy/systemd/rigged-royale-collector.service.in"
unit_name="rigged-royale-collector.service"
env_file="${repo_root}/.env"
run_user="${SUDO_USER:-$(id -un)}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer must run on the Linux Netcup server." >&2
  exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemd is required (systemctl was not found)." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install python3, python3-venv and python3-pip." >&2
  exit 1
fi
if [[ ! -f "${env_file}" ]]; then
  echo "Missing ${env_file}. Copy .env.example to .env and fill the secrets first." >&2
  exit 1
fi
if [[ ! -f "${repo_root}/seeds_season88.txt" ]]; then
  echo "Missing ${repo_root}/seeds_season88.txt." >&2
  exit 1
fi

for required_key in CR_API_TOKEN SUPABASE_URL SUPABASE_SECRET_KEY; do
  if ! grep -Eq "^${required_key}=.+$" "${env_file}"; then
    echo "${required_key} is missing or empty in ${env_file}." >&2
    exit 1
  fi
done

python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || {
  echo "Python 3.11 or newer is required." >&2
  exit 1
}

python3 -m venv "${repo_root}/.venv-collector"
"${repo_root}/.venv-collector/bin/python" -m pip install --upgrade pip setuptools wheel
"${repo_root}/.venv-collector/bin/python" -m pip install \
  "pyarrow>=17" \
  "psycopg[binary]>=3.2" \
  "python-dotenv>=1.0" \
  "PyYAML>=6.0" \
  "tqdm>=4.66"
"${repo_root}/.venv-collector/bin/python" -m pip install --no-deps --editable "${repo_root}"

mkdir -p "${repo_root}/data/season-88/raw" "${repo_root}/logs"
if [[ "$(id -u)" -eq 0 ]]; then
  chown -R "${run_user}:$(id -gn "${run_user}")" \
    "${repo_root}/.venv-collector" \
    "${repo_root}/data" \
    "${repo_root}/logs"
fi

unit_tmp="$(mktemp)"
trap 'rm -f "${unit_tmp}"' EXIT
sed \
  -e "s|__REPO_ROOT__|${repo_root}|g" \
  -e "s|__RUN_USER__|${run_user}|g" \
  "${unit_template}" > "${unit_tmp}"

if [[ "$(id -u)" -eq 0 ]]; then
  admin=()
else
  admin=(sudo)
fi
"${admin[@]}" install -m 0644 "${unit_tmp}" "/etc/systemd/system/${unit_name}"
"${admin[@]}" systemctl daemon-reload
"${admin[@]}" systemctl enable "${unit_name}"
# restart, not `enable --now`: on an update the old process would otherwise keep
# running the previous code (and the previous unit's environment) until its
# cycle ends. SIGINT lets it flush its partial shard first.
"${admin[@]}" systemctl restart "${unit_name}"

echo "Collector installed and started."
echo "Status: sudo systemctl status ${unit_name}"
echo "Logs:   sudo journalctl -u ${unit_name} -f"
