#!/usr/bin/env bash
# VisionLab 训练管理服务启动脚本
# 用法：
#   ./start.sh          生产模式（默认）：不启用热重载，训练任务不会因代码改动重启而中断
#   ./start.sh --dev    开发模式：启用 --reload，修改代码会自动重启服务（会中断正在运行的训练）
set -e
# readlink -f 解析软链接真实路径，保证通过全局命令 visionlab 执行时 SCRIPT_DIR 指向仓库目录
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$SCRIPT_DIR"

# 服务端口：优先读取安装时写入的 .visionlab_port，默认 80
PORT_FILE="$SCRIPT_DIR/.visionlab_port"
PORT="80"
if [ -f "$PORT_FILE" ]; then PORT="$(tr -d '[:space:]' < "$PORT_FILE")"; fi

MODE="prod"
for arg in "$@"; do
  case "$arg" in
    --dev) MODE="dev" ;;
  esac
done

if ! command -v python3 &>/dev/null; then
  echo "未检测到 python3，请先安装 Python 3.10+。"
  echo "Ubuntu/Debian:  sudo apt install -y python3 python3-pip python3-venv"
  echo "CentOS/RHEL:    sudo dnf install -y python3 python3-pip"
  exit 1
fi

# 服务已在运行时直接提示退出（避免端口占用报错 address already in use）
if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
  echo "VisionLab 服务已在运行：http://127.0.0.1:${PORT}"
  echo "如需重启，请先停止：cd \"$SCRIPT_DIR\" && ./stop.sh && ./start.sh"
  exit 0
fi

# 优先使用项目虚拟环境（避免 Ubuntu 24.04+ 的 PEP 668 限制）
# 注意：Web 服务运行在 venv；训练/转换子进程使用对应模型的 Conda 环境（见 server/training.py）
VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "未找到虚拟环境，正在创建并安装依赖…"
  python3 -m venv "$SCRIPT_DIR/venv" || {
    echo "创建虚拟环境失败，请先运行 bash install.sh"
    exit 1
  }
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements-server.txt
else
  "$VENV_PY" -m pip install -r requirements-server.txt
fi

if [ "$MODE" = "dev" ]; then
  echo "[开发模式] 已启用 --reload，修改代码会重启服务（正在运行的训练将被中断）。"
  exec "$VENV_PY" -m uvicorn server.main:app --host 0.0.0.0 --port "$PORT" --reload
fi

echo "[生产模式] 服务不会因热重载重启，训练任务保持不中断。"
exec "$VENV_PY" -m uvicorn server.main:app --host 0.0.0.0 --port "$PORT"
