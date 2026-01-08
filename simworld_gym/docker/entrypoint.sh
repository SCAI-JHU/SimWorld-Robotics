#!/bin/bash
set -euo pipefail

chown -R sim:sim "$WORKDIR"
gosu sim /userscript.sh &
ue_pid=$!

cleanup() {
  echo "Stopping background processes..."
  kill -TERM "$ue_pid" 2>/dev/null || true
  wait "$ue_pid" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

cd "$WORKDIR"
exec jupyter notebook --ip=0.0.0.0 --port="$JUPYTER_PORT" --allow-root --no-browser