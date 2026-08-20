#!/usr/bin/env bash
# VisionLab 训练管理服务启动脚本
# 用法：
#   ./start.sh          生产模式（默认）：不启用热重载，训练任务不会因代码改动重启而中断
#   ./start.sh --dev    开发模式：启用 --reload，修改代码会自动重启服务（会中断正在运行的训练）
set -e
cd "$(dirname "$0")"

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

python3 -m pip install -r requirements.txt

# 可选依赖：rknn-toolkit2（RKNN 转换需要，瑞芯微 NPU 格式）
# 仅支持 Linux（x86_64/aarch64）且 Python 3.8-3.12；安装失败不影响其他功能。
if ! python3 -c "from rknn.api import RKNN" &>/dev/null; then
  echo "[可选] 正在安装 rknn-toolkit2（RKNN 模型转换需要）…"
  if python3 -m pip install "rknn-toolkit2>=2.3.2"; then
    echo "[可选] rknn-toolkit2 安装成功"
  else
    echo "[警告] rknn-toolkit2 安装失败：RKNN 转换功能不可用（不影响训练与其他转换格式）"
    echo "        如需 RKNN 转换，请确认系统为 Linux x86_64/aarch64 且 Python 版本为 3.8-3.12"
  fi
fi

if [ "$MODE" = "dev" ]; then
  echo "[开发模式] 已启用 --reload，修改代码会重启服务（正在运行的训练将被中断）。"
  exec python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
fi

echo "[生产模式] 服务不会因热重载重启，训练任务保持不中断。"
exec python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000
