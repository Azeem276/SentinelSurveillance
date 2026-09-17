#!/usr/bin/env bash
# Build the dashboard and publish it to the directory nginx serves.
#
# nginx runs as www-data, which cannot traverse a private home directory, so
# the bundle is copied out of the repo rather than served from frontend/dist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT="${SENTINEL_WEB_ROOT:-/var/www/sentinel}"

cd "$REPO_ROOT/frontend"
npm run lint
npm run build

sudo mkdir -p "$WEB_ROOT"
sudo rsync -a --delete dist/ "$WEB_ROOT/"
sudo chown -R root:www-data "$WEB_ROOT"
sudo chmod -R u=rwX,g=rX,o= "$WEB_ROOT"
echo "published $(sudo du -sh "$WEB_ROOT" | cut -f1) to $WEB_ROOT"
