# Deployment (EC2 + GitHub Actions)

Every push/merge to `main` runs [.github/workflows/deploy.yml](../.github/workflows/deploy.yml), which SSHes into the
EC2 host and runs [scripts/deploy/deploy.sh](../scripts/deploy/deploy.sh) in `~/AI-Backend/VIRU-AI-HUB`:

1. Fetch `main` (using the workflow's short-lived token, so no credentials live on the server) and hard-reset to it.
   Untracked files (`.env`, uploads) are untouched; `history/`, `output/`, `enterpriseRAG/data` and the upload dirs
   are backed up and restored so server-written data wins.
2. `docker compose build` — Docker reuses cached layers, so `apt`/`pip install` only re-run when `Dockerfile` or
   `requirements.txt` change. New dependencies are picked up automatically.
3. `docker compose up -d` and poll `http://127.0.0.1:8000/` for up to 180 s. If it never comes up, the previous image is
   restored and the workflow fails.

EC2 stop/start: the container has `restart: unless-stopped` and Docker is enabled at boot, so the app comes back
by itself. Deploys only happen on push; a push while the instance is stopped fails, so re-run it from the Actions tab.

## One-time EC2 setup

```bash
# Docker Engine + Compose v2 (skip if `docker compose version` already works)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu        # log out and back in afterwards
sudo systemctl enable --now docker

# Stop whatever runs the app today (nohup/screen/systemd/cron @reboot) so port 8000 is free,
# then do the first deploy by hand:
cd ~/AI-Backend/VIRU-AI-HUB
bash scripts/deploy/deploy.sh         # after the workflow files are on main; or let the first push do it
```

Keep `~/AI-Backend/VIRU-AI-HUB/.env` on the server; it is never in git.

## GitHub secrets

Repo → Settings → Secrets and variables → Actions:

| Secret | Value |
| --- | --- |
| `EC2_HOST` | Public IP or DNS of the instance |
| `EC2_SSH_KEY` | Full contents of the `.pem` private key |
| `EC2_USER` | Optional, default `ubuntu` |
| `EC2_PORT` | Optional, default `22` |

The public IP changes on every stop/start unless you attach an **Elastic IP** (free while attached to a running
instance). Without one, update `EC2_HOST` after each restart.

The security group must allow inbound SSH (22) from GitHub-hosted runners, whose IPs vary, so in practice
`0.0.0.0/0` with key-only auth.

## Operations

```bash
docker compose logs -f web            # logs
docker compose restart web            # restart without redeploying
bash scripts/deploy/deploy.sh         # manual redeploy (uses the server's own git remote)
```
