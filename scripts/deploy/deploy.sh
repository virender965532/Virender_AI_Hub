#!/usr/bin/env bash
# Server-side deploy for VIRU-AI-HUB on EC2. Invoked by .github/workflows/deploy.yml;
# can also be run by hand on the server:  bash scripts/deploy/deploy.sh
#
# Steps: sync code to origin/main (keeping server-written data) -> rebuild image
# (pip/apt layers only re-run when requirements.txt / Dockerfile change) ->
# restart container -> health check -> roll back to the previous image on failure ->
# reclaim disk (old images, stopped containers, build cache above a cap).
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AI-Backend/VIRU-AI-HUB}"
BRANCH="${BRANCH:-main}"
REPO="${REPO:-}"                       # owner/name, set by the workflow
IMAGE="viru-ai-hub"
APP_PORT="${APP_PORT:-8000}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:$APP_PORT/}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
# Build cache kept so unchanged pip/apt layers stay cached; everything above this is freed.
BUILD_CACHE_KEEP="${BUILD_CACHE_KEEP:-5GB}"
# Below this much free disk, wipe the whole build cache before building.
MIN_FREE_GB="${MIN_FREE_GB:-8}"
# 1 = also remove unused images that are not this app's (other projects, old base images).
PRUNE_OTHER_IMAGES="${PRUNE_OTHER_IMAGES:-1}"

# The workflow sends its short-lived GITHUB_TOKEN as the first stdin line so it
# never appears in argv or on disk. Then detach stdin so nothing below eats it.
GIT_TOKEN=""
if [ ! -t 0 ]; then read -r GIT_TOKEN || true; fi
exec </dev/null

# Tracked in git but written by the app at runtime: server copy wins.
RUNTIME_DIRS=(history output enterpriseRAG/data dynamicRAG/uploads enterpriseRAG/uploads)

log() { printf '\n==> %s\n' "$*"; }

free_gb() {
  local dir=/var/lib/docker
  [ -d "$dir" ] || dir=/
  df -Pk "$dir" | awk 'NR==2 {print int($4 / 1048576)}'
}

prune_build_cache() {
  # --reserved-space is the Docker 28+ name for --keep-storage.
  docker builder prune -f --reserved-space "$1" >/dev/null 2>&1 \
    || docker builder prune -f --keep-storage "$1" >/dev/null 2>&1 || true
}

cleanup() {
  # Never touches volumes or the app's bind-mounted data.
  docker container prune -f >/dev/null || true
  # Old builds: re-tagging leaves the image before "previous" untagged (dangling).
  docker image prune -f >/dev/null || true
  if [ "$PRUNE_OTHER_IMAGES" = "1" ]; then
    # Unused images without this app's label; latest/previous carry it and are kept.
    docker image prune -af --filter "label!=app=$IMAGE" >/dev/null || true
  fi
  docker network prune -f >/dev/null || true
  prune_build_cache "$BUILD_CACHE_KEEP"
}

cd "$APP_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed on this host. See docs/DEPLOYMENT.md (one-time setup)." >&2
  exit 1
fi
# Docker must start on boot so the container comes back after an EC2 stop/start.
sudo systemctl enable --now docker >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
log "Backing up runtime data"
BACKUP_DIR="$(mktemp -d)"
trap 'rm -rf "$BACKUP_DIR"' EXIT
for d in "${RUNTIME_DIRS[@]}"; do
  if [ -d "$d" ]; then
    mkdir -p "$BACKUP_DIR/$d"
    cp -a "$d/." "$BACKUP_DIR/$d/"
  fi
done

log "Fetching $BRANCH"
if [ -n "$GIT_TOKEN" ] && [ -n "$REPO" ]; then
  AUTH="$(printf 'x-access-token:%s' "$GIT_TOKEN" | base64 -w0)"
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $AUTH" \
    fetch --prune "https://github.com/$REPO.git" "$BRANCH"
else
  git fetch --prune origin "$BRANCH"
fi
OLD_REV="$(git rev-parse --short HEAD 2>/dev/null || echo none)"
# Untracked files (.env, uploads, Profile additions) are left alone by reset.
git checkout -q -f -B "$BRANCH" FETCH_HEAD
git reset -q --hard FETCH_HEAD
NEW_REV="$(git rev-parse --short HEAD)"
echo "$OLD_REV -> $NEW_REV"

log "Restoring runtime data"
for d in "${RUNTIME_DIRS[@]}"; do
  if [ -d "$BACKUP_DIR/$d" ]; then
    mkdir -p "$d"
    cp -a "$BACKUP_DIR/$d/." "$d/"
  fi
done
mkdir -p scraper_debug config Profile "${RUNTIME_DIRS[@]}"

if [ ! -f .env ]; then
  echo ".env is missing in $APP_DIR — copy .env.example and fill it in." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
log "Building image"
FREE="$(free_gb)"
echo "Free disk: ${FREE} GB"
if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
  echo "Low disk space, clearing unused images and the whole build cache first"
  cleanup
  docker builder prune -af >/dev/null || true
  echo "Free disk now: $(free_gb) GB"
fi
HAD_PREVIOUS=0
if docker image inspect "$IMAGE:latest" >/dev/null 2>&1; then
  docker tag "$IMAGE:latest" "$IMAGE:previous"
  HAD_PREVIOUS=1
fi
docker compose build web

log "Restarting container"
# A container from another compose project or a manual `docker run` (e.g. an older
# copy of this app) holding the port makes `up` fail with "port is already allocated".
for id in $(docker ps -q --filter "publish=$APP_PORT"); do
  project="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$id")"
  if [ "$project" != "$IMAGE" ]; then
    echo "Removing container $(docker inspect -f '{{.Name}}' "$id") (project '${project:-none}') holding port $APP_PORT"
    docker rm -f "$id" >/dev/null
  fi
done

healthy=0
if docker compose up -d --remove-orphans web; then
  log "Health check: $HEALTH_URL"
  for ((i = 0; i < HEALTH_TIMEOUT; i += 5)); do
    if curl -fsS -o /dev/null --max-time 5 "$HEALTH_URL"; then healthy=1; break; fi
    sleep 5
  done
else
  echo "docker compose up failed" >&2
  if sudo ss -ltnp "sport = :$APP_PORT" 2>/dev/null | grep -q LISTEN; then
    echo "Port $APP_PORT is held by a non-Docker process:" >&2
    sudo ss -ltnp "sport = :$APP_PORT" >&2 || true
  fi
fi

if [ "$healthy" -ne 1 ]; then
  echo "App did not become healthy within ${HEALTH_TIMEOUT}s. Recent logs:" >&2
  docker compose logs --tail 80 web >&2 || true
  if [ "$HAD_PREVIOUS" -eq 1 ]; then
    log "Rolling back to previous image"
    docker tag "$IMAGE:previous" "$IMAGE:latest"
    docker compose up -d --no-build --force-recreate web
  fi
  exit 1
fi

log "Reclaiming disk"
cleanup
git gc --auto --quiet || true
docker system df
echo "Free disk: $(free_gb) GB"

log "Deployed $NEW_REV"
docker compose ps
