#!/usr/bin/env bash
# Server-side deploy for VIRU-AI-HUB on EC2. Invoked by .github/workflows/deploy.yml;
# can also be run by hand on the server:  bash scripts/deploy/deploy.sh
#
# Steps: sync code to origin/main (keeping server-written data) -> rebuild image
# (pip/apt layers only re-run when requirements.txt / Dockerfile change) ->
# restart container -> health check -> roll back to the previous image on failure.
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AI-Backend/VIRU-AI-HUB}"
BRANCH="${BRANCH:-main}"
REPO="${REPO:-}"                       # owner/name, set by the workflow
IMAGE="viru-ai-hub"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"

# The workflow sends its short-lived GITHUB_TOKEN as the first stdin line so it
# never appears in argv or on disk. Then detach stdin so nothing below eats it.
GIT_TOKEN=""
if [ ! -t 0 ]; then read -r GIT_TOKEN || true; fi
exec </dev/null

# Tracked in git but written by the app at runtime: server copy wins.
RUNTIME_DIRS=(history output enterpriseRAG/data dynamicRAG/uploads enterpriseRAG/uploads)

log() { printf '\n==> %s\n' "$*"; }

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
HAD_PREVIOUS=0
if docker image inspect "$IMAGE:latest" >/dev/null 2>&1; then
  docker tag "$IMAGE:latest" "$IMAGE:previous"
  HAD_PREVIOUS=1
fi
docker compose build web

log "Restarting container"
docker compose up -d --remove-orphans web

log "Health check: $HEALTH_URL"
healthy=0
for ((i = 0; i < HEALTH_TIMEOUT; i += 5)); do
  if curl -fsS -o /dev/null --max-time 5 "$HEALTH_URL"; then healthy=1; break; fi
  sleep 5
done

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

log "Cleaning up dangling images"
docker image prune -f >/dev/null || true

log "Deployed $NEW_REV"
docker compose ps
