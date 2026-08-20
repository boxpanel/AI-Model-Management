#!/usr/bin/env bash
# ============================================================
# VisionLab 一键安装脚本：安装程序所需的全部依赖与工具
# 用法：
#   ./install.sh              安装全部（系统工具 + Python 依赖 + RKNN 转换依赖）
#   ./install.sh --skip-rknn  跳过 rknn-toolkit2 安装
#   ./install.sh --no-sudo    不使用 sudo（跳过系统级依赖，仅安装 Python 依赖）
# ============================================================
set -e
cd "$(dirname "$0")"

SKIP_RKNN=0
NO_SUDO=0
for arg in "$@"; do
  case "$arg" in
    --skip-rknn) SKIP_RKNN=1 ;;
    --no-sudo)   NO_SUDO=1 ;;
    *) echo "未知参数：$arg（可用：--skip-rknn / --no-sudo）" ;;
  esac
done

echo "=========================================="
echo " VisionLab 依赖安装"
echo "=========================================="

# ---------- 1. 检测 Python ----------
if ! command -v python3 &>/dev/null; then
  echo "[错误] 未检测到 python3，请先安装 Python 3.10-3.12："
  echo "  Ubuntu/Debian: sudo apt install -y python3 python3-pip python3-venv"
  echo "  CentOS/RHEL:   sudo dnf install -y python3 python3-pip"
  exit 1
fi
echo "[1/5] Python: $(python3 --version)（架构 $(uname -m)）"

# ---------- 2. 安装系统级依赖（工具库） ----------
if [ "$NO_SUDO" = "0" ]; then
  if command -v apt-get &>/dev/null; then
    echo "[2/5] 安装系统依赖（apt）…"
    sudo apt-get update
    sudo apt-get install -y \
      python3-dev python3-pip python3-venv \
      libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx \
      libprotobuf-dev gcc g++ curl wget git
  elif command -v dnf &>/dev/null; then
    echo "[2/5] 安装系统依赖（dnf）…"
    sudo dnf install -y \
      python3-devel python3-pip \
      libxslt-devel zlib-devel glib2-devel \
      libSM libXext libXrender mesa-libGL \
      protobuf-devel gcc gcc-c++ curl wget git
  else
    echo "[2/5][警告] 未识别的包管理器，请手动安装系统依赖"
  fi
else
  echo "[2/5] 已跳过系统依赖安装（--no-sudo）"
fi

# ---------- 3. 安装 Python 依赖 ----------
echo "[3/5] 安装 Python 依赖（requirements.txt）…"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# ---------- 4. RKNN 转换依赖（可选） ----------
if [ "$SKIP_RKNN" = "1" ]; then
  echo "[4/5] 已跳过 rknn-toolkit2（--skip-rknn）"
elif python3 -c "from rknn.api import RKNN" &>/dev/null; then
  echo "[4/5] rknn-toolkit2 已安装"
else
  echo "[4/5] 安装 rknn-toolkit2（RKNN 转换需要，仅 Linux x86_64/aarch64 + Python 3.8-3.12）…"
  if python3 -m pip install "rknn-toolkit2>=2.3.2"; then
    echo "[4/5] rknn-toolkit2 安装成功"
  else
    echo "[4/5][警告] rknn-toolkit2 安装失败：RKNN 转换不可用（不影响训练与其他转换格式）"
  fi
fi

# ---------- 5. GPU / CUDA 环境检测 ----------
echo "[5/5] 环境检测…"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "  NVIDIA 驱动: 已检测到（GPU 训练可用）"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 8 | sed 's/^/    /'
else
  echo "  NVIDIA 驱动: 未检测到"
  echo "  [提示] 如需 GPU 训练，请安装 NVIDIA 驱动后重启系统；"
  echo "         未安装驱动时训练将使用 CPU（device=cpu）。"
fi

echo ""
echo "=========================================="
echo " 安装完成！"
echo " 启动服务：      ./start.sh"
echo " 开发模式启动：  ./start.sh --dev"
echo "=========================================="
