#!/usr/bin/env bash
# ============================================================
# VisionLab 一键安装脚本：安装程序所需的全部依赖与工具
# 支持两种执行方式：
#   1. 远程一键（推荐）：curl -fsSL <raw-url> | bash
#      - 自动克隆/更新仓库到 ~/AI-Model-Management（目录已存在时自动 git pull，不会报错）
#   2. 仓库内执行：bash install.sh [--skip-rknn] [--no-sudo]
#
# 安装内容：
#   - Web 服务依赖 → 项目 venv 虚拟环境（requirements-server.txt）
#   - Miniconda（系统无 conda 时自动安装）
#   - 训练 Conda 环境 YOLOv11 / YOLOv8 / YOLOv5（各自安装 requirements.txt，Python 3.11）
#   - rknn-toolkit2 → 各训练 Conda 环境（可选）
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
echo "[1/6] Python: $(python3 --version)（架构 $(uname -m)）"

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
    echo "[2/6] 安装系统依赖（apt）…"
    sudo apt-get update
    install_apt_pkgs \
      python3-dev python3-pip python3-venv \
      libxslt1-dev zlib1g zlib1g-dev \
      libglib2.0-0 libglib2.0-0t64 libsm6 libgl1 \
      libprotobuf-dev gcc g++ curl wget git
  elif command -v dnf &>/dev/null; then
    echo "[2/6] 安装系统依赖（dnf）…"
    sudo dnf install -y \
      python3-devel python3-pip \
      libxslt-devel zlib-devel glib2-devel \
      libSM libXext libXrender mesa-libGL \
      protobuf-devel gcc gcc-c++ curl wget git
  else
    echo "[2/6][警告] 未识别的包管理器，请手动安装系统依赖"
  fi
else
  echo "[2/6] 已跳过系统依赖安装（--no-sudo）"
fi

# ---------- 3. 安装 Miniconda（系统无 conda 时） ----------
CONDA_BIN="$(command -v conda || true)"
if [ -z "$CONDA_BIN" ]; then
  MINICONDA_DIR="$HOME/miniconda3"
  if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then
    echo "[3/6] 未检测到 conda，正在安装 Miniconda 到 $MINICONDA_DIR …"
    ARCH="$(uname -m)"
    if [ "$ARCH" = "x86_64" ]; then
      INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
      INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
    else
      echo "[3/6][警告] 不支持的架构 $ARCH，跳过 Miniconda 安装（训练将使用 venv）"
      INSTALLER=""
    fi
    if [ -n "$INSTALLER" ]; then
      curl -fsSL "https://repo.anaconda.com/miniconda/$INSTALLER" -o /tmp/miniconda.sh
      bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
      rm -f /tmp/miniconda.sh
      CONDA_BIN="$MINICONDA_DIR/bin/conda"
    fi
  else
    CONDA_BIN="$MINICONDA_DIR/bin/conda"
  fi
fi
if [ -n "$CONDA_BIN" ]; then
  echo "[3/6] conda: $("$CONDA_BIN" --version 2>/dev/null || echo '已安装')"
  # 接受 Anaconda 默认渠道的服务条款（2025 年起访问 repo.anaconda.com 需接受）
  "$CONDA_BIN" tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/main" >/dev/null 2>&1 || true
  "$CONDA_BIN" tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/r" >/dev/null 2>&1 || true
else
  echo "[3/6][警告] conda 不可用，训练环境将回退到 venv"
fi

# ---------- 4. 创建并配置训练 Conda 环境 ----------
MODEL_ENVS=("YOLOv11" "YOLOv8" "YOLOv5")
echo "[4/6] 创建训练 Conda 环境（${MODEL_ENVS[*]}，Python 3.11）…"
for ENV_NAME in "${MODEL_ENVS[@]}"; do
  if [ -z "$CONDA_BIN" ]; then
    break
  fi
  if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  - 创建环境 $ENV_NAME …"
    # 使用 conda-forge 渠道（--override-channels 忽略默认渠道），避免 Anaconda ToS 限制
    "$CONDA_BIN" create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.11
  else
    echo "  - 环境 $ENV_NAME 已存在"
  fi
  echo "  - 安装依赖到 $ENV_NAME …"
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r requirements.txt
  if [ "$SKIP_RKNN" = "0" ]; then
    "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install "rknn-toolkit2>=2.3.2" \
      || echo "  [警告] $ENV_NAME 环境 rknn-toolkit2 安装失败（RKNN 转换不可用，不影响训练）"
  fi
done

# ---------- 5. Web 服务虚拟环境（venv） ----------
# 使用 venv 避免 Ubuntu 24.04+（PEP 668）禁止系统级 pip 安装的问题
echo "[5/6] 创建 Web 服务虚拟环境并安装服务依赖…"
VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  python3 -m venv "$SCRIPT_DIR/venv" || {
    echo "[错误] 创建虚拟环境失败，请确认已安装 python3-venv：sudo apt install -y python3-venv"
    exit 1
  }
fi
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements-server.txt

# ---------- 6. GPU / CUDA 环境检测 ----------
echo "[6/6] 环境检测…"
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
