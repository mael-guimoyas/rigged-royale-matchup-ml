#!/usr/bin/env bash
#
# Rebuild the CPU serving image in place on the host that runs it.
#
# The netcup deployment reads `rigged-model:latest` from the local docker image
# store (see compose.yml in riggedroyale/migration-cloudrun-hetzner), so nothing
# needs a registry: build here, recreate the container, done.
#
# The point of running this is that the deployed image predates the vectorised
# encoder. The row encoder was ~40% of CPU time and is now 22-31x faster, so the
# same hardware should serve materially more rows per second.
#
# Usage:
#   bash scripts/build_cpu_image.sh              # build only, prints next steps
#   DEPLOY=1 bash scripts/build_cpu_image.sh     # build and recreate the container
#
# Optional:
#   IMAGE=...            image tag to build (default rigged-model:latest)
#   MODEL_CONTAINER=...  running container to take the checkpoint from
#   COMPOSE_DIR=...      directory holding compose.yml, if autodetection fails
#
# The previous image is retagged rigged-model:previous before anything is
# replaced, so a rollback is one command and never needs a rebuild.

set -euo pipefail

IMAGE="${IMAGE:-rigged-model:latest}"
BACKUP_IMAGE="${BACKUP_IMAGE:-rigged-model:previous}"
MODEL_CONTAINER="${MODEL_CONTAINER:-rigged-model}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

say() { printf '\n==> %s\n' "$1"; }
die() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "cannot reach the docker daemon"
[ -f Dockerfile ] || die "run this from the repository root (no ./Dockerfile here)"

free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "only ${free_gb} GB free, need about ${MIN_FREE_GB} GB"

# --- checkpoint --------------------------------------------------------------
# artifacts/ is gitignored, so a clone has no model. Take the one currently in
# production: that keeps the rebuilt image serving the exact same weights, which
# is what makes this a pure performance change rather than a model change.

mkdir -p artifacts
if [ ! -s artifacts/matchup-model.pt ]; then
  docker inspect "$MODEL_CONTAINER" >/dev/null 2>&1 \
    || die "container '$MODEL_CONTAINER' not found; set MODEL_CONTAINER or place the checkpoint yourself"
  say "Extracting the live checkpoint from '$MODEL_CONTAINER'"
  docker cp "$MODEL_CONTAINER:/app/artifacts/matchup-model.pt" artifacts/matchup-model.pt
fi
before_sha=$(sha256sum artifacts/matchup-model.pt | cut -d' ' -f1)
printf 'checkpoint sha256: %s\n' "$before_sha"

# --- keep a way back ---------------------------------------------------------

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "Tagging the current image as $BACKUP_IMAGE"
  docker tag "$IMAGE" "$BACKUP_IMAGE"
else
  printf 'no existing %s to back up\n' "$IMAGE"
fi

# --- build -------------------------------------------------------------------

say "Building $IMAGE from the CPU Dockerfile"
docker build -f Dockerfile -t "$IMAGE" .

# --- deploy ------------------------------------------------------------------

compose_dir="${COMPOSE_DIR:-$(docker inspect "$MODEL_CONTAINER" \
  --format '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' 2>/dev/null || true)}"

if [ "${DEPLOY:-0}" != "1" ]; then
  say "Built, not deployed (DEPLOY=1 to recreate the container)"
  cat <<NEXT

To deploy:
  cd ${compose_dir:-<directory holding compose.yml>}
  docker compose up -d --force-recreate model
  curl -s localhost:8080/health

Expect "device":"cpu" in /health -- the field only exists in the new image, so
its presence is how you confirm the recreate actually took.

To roll back:
  docker tag $BACKUP_IMAGE $IMAGE
  docker compose up -d --force-recreate model
NEXT
  exit 0
fi

[ -n "$compose_dir" ] || die "could not find the compose directory; set COMPOSE_DIR"
say "Recreating the model container from $compose_dir"
( cd "$compose_dir" && docker compose up -d --force-recreate model )

sleep 3
say "Health after recreate"
curl -s --max-time 20 localhost:8080/health || printf '(no answer yet; give it a few seconds)\n'
cat <<NEXT


To roll back:
  docker tag $BACKUP_IMAGE $IMAGE
  cd $compose_dir && docker compose up -d --force-recreate model
NEXT
