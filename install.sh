#!/usr/bin/env bash
# ============================================================
# VisionLab 一键安装脚本：安装程序所需的全部依赖与工具
# 支持两种执行方式：
#   1. 远程一键（推荐）：curl -fsSL <raw-url> | bash
#      - 自动克隆/更新仓库到 ~/AI-Model-Management（目录已存在时自动 git pull，不会报错）
#   2. 仓库内执行：bash install.sh [--skip-rknn] [--no-sudo]
# ============================================================
set -e

# ---------- 自引导：在仓库外执行时，先获取/更新仓库代码 ----------
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd || pwd)"
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
  REPO_DIR="${VISIONLAB_DIR:-$HOME/AI-Model-Management}"
  if [ -d "$REPO_DIR/.git" ]; then
    echo "[引导] 检测到已存在的仓库 $REPO_DIR，正在拉取最新代码…"
    git -C "$REPO_DIR" pull --ff-only
  else
    echo "[引导] 正在克隆仓库到 $REPO_DIR …"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone https://github.com/boxpanel/AI-Model-Management.git "$REPO_DIR"
  fi
  echo "[引导] 继续执行安装…"
  exec bash "$REPO_DIR/install.sh" "$@"
fi
cd "$SCRIPT_DIR"

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
# apt 包名在不同 Ubuntu 版本有差异（如 24.04 移除 libgl1-mesa-glx，改为 libgl1），
# 安装前逐个检查是否有安装候选，不可用的包自动跳过，避免整体安装失败。
install_apt_pkgs() {
  local pkgs=()
  for pkg in "$@"; do
    if apt-cache policy "$pkg" 2>/dev/null | grep -q "^  Candidate: ."; then
      pkgs+=("$pkg")
    else
      echo "  [跳过] 当前系统无可用包：$pkg"
    fi
  done
  if [ "${#pkgs[@]}" -gt 0 ]; then
    sudo apt-get install -y "${pkgs[@]}"
  fi
}
if [ "$NO_SUDO" = "0" ]; then
  if command -v apt-get &>/dev/null; then
    echo "[2/5] 安装系统依赖（apt）…"
    sudo apt-get update
    install_apt_pkgs \
      python3-dev python3-pip python3-venv \
      libxslt1-dev zlib1g zlib1g-dev \
      libglib2.0-0 libglib2.0-0t64 libsm6 libgl1 \
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
