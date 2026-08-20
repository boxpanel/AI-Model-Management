#!/usr/bin/env bash
# VisionLab 停止服务
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PORT_FILE="$SCRIPT_DIR/.visionlab_port"
PORT="80"
if [ -f "$PORT_FILE" ]; then PORT="$(tr -d '[:space:]' < "$PORT_FILE")"; fi

PIDS="$(pgrep -f 'uvicorn server.main:app' || true)"
if [ -n "$PIDS" ]; then
  echo "正在停止 VisionLab 服务（PID: $(echo "$PIDS" | tr '\n' ' ')）…"
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null || true
  sleep 1
  PIDS_LEFT="$(pgrep -f 'uvicorn server.main:app' || true)"
  if [ -n "$PIDS_LEFT" ]; then
    # shellcheck disable=SC2086
    kill -9 $PIDS_LEFT 2>/dev/null || true
  fi
  echo "服务已停止。"
else
  echo "VisionLab 服务未在运行。"
fi
