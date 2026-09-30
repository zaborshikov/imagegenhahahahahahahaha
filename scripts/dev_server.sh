#!/usr/bin/env bash
# Dev helper: (re)start the server detached from the calling shell on a chosen GPU.
# Usage: scripts/dev_server.sh [gpu_index]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-server.log}"

pkill -f "$PY run.py" 2>/dev/null || true
sleep 1
CUDA_VISIBLE_DEVICES="$GPU" setsid "$PY" run.py > "$LOG" 2>&1 < /dev/null &
disown
echo "started on GPU $GPU (pid $!), log: $LOG"
