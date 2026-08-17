#!/usr/bin/env bash
#
# Build and publish the CUDA serving image, from a machine with real bandwidth.
#
# Meant to run on the netcup host rather than a laptop: the image is several
# gigabytes, `docker push` is an upload, and a residential upstream makes that
# painful. The host already runs the CPU service, so it also already has the
# checkpoint -- this script lifts it out of the running container instead of
# asking anyone to copy it around.
#
# It does not touch the running containers. `docker cp` reads from one; the
# build only consumes CPU, disk and network.
#
# Usage:
#   IMAGE=docker.io/<user>/rigged-model-gpu:latest bash scripts/build_gpu_image.sh
#
# Optional:
#   WORKDIR=...          where to clone/refresh the repo (default ~/rigged-gpu-build)
#   MODEL_CONTAINER=...  container holding the checkpoint (default rigged-model)
#   EXPECTED_SHA=...     abort unless the checkpoint matches this sha256
#   SKIP_PUSH=1          build only, do not publish

set -euo pipefail

IMAGE="${IMAGE:-}"
WORKDIR="${WORKDIR:-$HOME/rigged-gpu-build}"
MODEL_CONTAINER="${MODEL_CONTAINER:-rigged-model}"
REPO_URL="${REPO_URL:-https://github.com/mael-guimoyas/rigged-royale-matchup-ml.git}"
EXPECTED_SHA="${EXPECTED_SHA:-}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

# The checkpoint in production on 2026-08-17, for reference. Deliberately not
# the default for EXPECTED_SHA: hard-coding it here would make this script lie
# after the next retrain.
#   be03d4b45227e9d54a6fb5f96cee1cd87b6855b2b427e87ee681a0b1f2ffbb5e
#   -> model_version v5-be03d4b45227

say() { printf '\n==> %s\n' "$1"; }
die() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------

[ -n "$IMAGE" ] || die "Set IMAGE, e.g. IMAGE=docker.io/<user>/rigged-model-gpu:latest"
command -v docker >/dev/null || die "docker not found"
command -v git >/dev/null || die "git not found"
docker info >/dev/null 2>&1 || die "cannot reach the docker daemon (try sudo, or add your user to the docker group)"

free_gb=$(df -Pk "$(dirname "$WORKDIR")" | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  die "only ${free_gb} GB free; the CUDA image and its build layers need about ${MIN_FREE_GB} GB"
fi
say "Preflight OK (${free_gb} GB free, docker reachable)"

# --- source ------------------------------------------------------------------

if [ -d "$WORKDIR/.git" ]; then
  say "Refreshing $WORKDIR"
  git -C "$WORKDIR" fetch --quiet origin
  git -C "$WORKDIR" checkout --quiet main
  git -C "$WORKDIR" reset --hard --quiet origin/main
else
  say "Cloning into $WORKDIR"
  git clone --quiet "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
printf 'commit: %s\n' "$(git log -1 --oneline)"

# --- checkpoint --------------------------------------------------------------
#
# artifacts/ is gitignored, so the clone has no model. The running CPU container
# has it baked in at /app/artifacts/matchup-model.pt, which is by construction
# the checkpoint currently serving production -- exactly the one the GPU image
# should carry, so the two deployments agree.

mkdir -p artifacts
if [ -s artifacts/matchup-model.pt ]; then
  say "Reusing the checkpoint already in $WORKDIR/artifacts"
else
  docker inspect "$MODEL_CONTAINER" >/dev/null 2>&1 \
    || die "container '$MODEL_CONTAINER' not found; set MODEL_CONTAINER, or copy the checkpoint to $WORKDIR/artifacts/matchup-model.pt yourself"
  say "Extracting the checkpoint from container '$MODEL_CONTAINER' (read-only)"
  docker cp "$MODEL_CONTAINER:/app/artifacts/matchup-model.pt" artifacts/matchup-model.pt
fi

[ -s artifacts/matchup-model.pt ] || die "artifacts/matchup-model.pt is missing or empty"
actual_sha=$(sha256sum artifacts/matchup-model.pt | cut -d' ' -f1)
printf 'checkpoint sha256: %s\n' "$actual_sha"
printf 'implies model_version: v<feature_version>-%s\n' "${actual_sha:0:12}"
if [ -n "$EXPECTED_SHA" ] && [ "$actual_sha" != "$EXPECTED_SHA" ]; then
  die "checkpoint sha256 does not match EXPECTED_SHA ($EXPECTED_SHA)"
fi

# --- build -------------------------------------------------------------------

say "Building $IMAGE (this pulls a few GB of CUDA wheels; expect several minutes)"
docker build -f Dockerfile.gpu -t "$IMAGE" .

size=$(docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.1f", $1/1024/1024/1024}')
say "Built $IMAGE (${size} GB)"

# --- publish -----------------------------------------------------------------

if [ "${SKIP_PUSH:-0}" = "1" ]; then
  say "SKIP_PUSH=1, stopping before publish"
  exit 0
fi

say "Pushing $IMAGE"
docker push "$IMAGE" || die "push failed -- run 'docker login <registry>' first, and check the repository exists"

digest=$(docker image inspect "$IMAGE" --format '{{range .RepoDigests}}{{.}}{{end}}' 2>/dev/null || true)
say "Done"
printf 'image:  %s\n' "$IMAGE"
[ -n "$digest" ] && printf 'digest: %s\n' "$digest"
cat <<'NEXT'

Next, in the RunPod endpoint form (see RUNPOD_SERVING.md):
  type          Load balancer          health check   /ping
  HTTP port     8080                   container disk 20 GB
  env           PORT=8080  MODEL_DEVICE=auto  MODEL_WARMUP=1
                MAX_BATCH_REQUESTS=2048  MODEL_NAME=symmetric-matchup

Then check the endpoint reports the same model_version as
https://model.riggedroyale.com/health before pointing the site at it.
NEXT
