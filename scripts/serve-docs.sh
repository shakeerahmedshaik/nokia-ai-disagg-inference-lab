#!/usr/bin/env bash
# Serve the interactive walkthrough (docs/index.html) locally. Usage: scripts/serve-docs.sh [port=8000]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PORT=${1:-8000}
echo "Walkthrough: http://localhost:$PORT  (Ctrl-C to stop)"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 -d "$HERE/docs"
