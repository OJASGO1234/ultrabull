#!/bin/bash
# Server-side deploy for bull.littpad.dev — run by the CD webhook on push to
# main (or manually: bash /home/ultrabull/deploy.sh). Assumes /home/ultrabull
# is a git clone; .env and data_cache/ are git-ignored so reset never touches
# them.
set -euo pipefail
exec 200>/tmp/ultrabull-deploy.lock
flock -n 200 || { echo "deploy already running"; exit 0; }

cd /home/ultrabull
echo "== $(date -Is) deploying =="
git fetch origin main
git reset --hard origin/main

docker build -t ultrabull .
docker rm -f ultrabull 2>/dev/null || true
docker run -d --name ultrabull --restart unless-stopped \
  -p 127.0.0.1:8010:8000 \
  --env-file /home/ultrabull/.env \
  -e ULTRABULL_AUTH=1 \
  -v /home/ultrabull/data_cache:/app/data_cache \
  ultrabull

cd frontend
npm ci --no-audit --no-fund
npm run build

echo "== deployed $(git rev-parse --short HEAD) at $(date -Is) =="
