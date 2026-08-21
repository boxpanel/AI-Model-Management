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

# 兼容所有 locale：强制 C locale，保证 apt-cache policy / conda / lspci 等输出为英文，
# 避免中文、日文等系统环境下英文关键字解析失败（如 apt 包可用性检测、GPU 检测等）
export LC_ALL=C
export LANG=C
export LANGUAGE=C

# ---------- 自引导：在仓库外执行时，先获取/更新仓库代码 ----------
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd || pwd)"
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
  REPO_DIR="${VISIONLAB_DIR:-$HOME/AI-Model-Management}"
  if [ -d "$REPO_DIR/.git" ]; then
    echo "[引导] 检测到已存在的仓库 $REPO_DIR，正在拉取最新代码…"
    git -C "$REPO_DIR" pull --ff-only
  else
    # 引导克隆依赖 git：缺失时自动安装（root 直接装，非 root 尝试 sudo）
    if ! command -v git &>/dev/null; then
      echo "[引导] 未检测到 git，正在自动安装…"
      GIT_INSTALL_OK=0
      if command -v apt-get &>/dev/null; then
        if [ "$(id -u)" -eq 0 ]; then apt-get update -y && apt-get install -y git && GIT_INSTALL_OK=1
        elif command -v sudo &>/dev/null; then sudo apt-get update -y && sudo apt-get install -y git && GIT_INSTALL_OK=1
        fi
      elif command -v dnf &>/dev/null; then
        if [ "$(id -u)" -eq 0 ]; then dnf install -y git && GIT_INSTALL_OK=1
        elif command -v sudo &>/dev/null; then sudo dnf install -y git && GIT_INSTALL_OK=1
        fi
      fi
      if [ "$GIT_INSTALL_OK" != "1" ]; then
        echo "[引导][错误] git 自动安装失败，请先手动安装后重新执行："
        echo "  sudo apt-get update -y && sudo apt-get install -y git"
        exit 1
      fi
      echo "[引导] git 安装完成"
    fi
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
DRIVER_AUTO=1
for arg in "$@"; do
  case "$arg" in
    --skip-rknn) SKIP_RKNN=1 ;;
    --no-sudo)   NO_SUDO=1 ;;
    --no-driver) DRIVER_AUTO=0 ;;
    *) echo "未知参数：$arg（可用：--skip-rknn / --no-sudo / --no-driver）" ;;
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
    # LC_ALL=C 强制英文输出，避免中文 locale 下 apt-cache policy 输出"候选"导致误判无可用包
    if LC_ALL=C apt-cache policy "$pkg" 2>/dev/null | grep -q "^  Candidate: ."; then
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
  # 将 conda 加入用户 shell PATH（新登录终端生效），便于手动管理环境
  "$CONDA_BIN" init bash >/dev/null 2>&1 || true
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
# venv 缺失或无 pip 时重建（兼容之前失败残留的残缺环境、python3-venv 缺失等情况）
if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "  创建虚拟环境（已清理旧的残缺环境）…"
  rm -rf "$SCRIPT_DIR/venv"
  python3 -m venv "$SCRIPT_DIR/venv" || {
    echo "[错误] 创建虚拟环境失败，请确认已安装 python3-venv：sudo apt install -y python3-venv"
    exit 1
  }
fi
# 兜底：venv 仍无 pip（如 ensurepip 未随 venv 生成）时用 ensurepip 安装
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "  虚拟环境缺少 pip，正在通过 ensurepip 安装…"
  "$VENV_PY" -m ensurepip --upgrade || {
    echo "[错误] venv 缺少 pip，请先安装系统 python3-venv 后重试：sudo apt install -y python3-venv"
    exit 1
  }
fi
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements-server.txt

# ---------- 创建默认工作目录 ----------
# 训练输出 / 数据集配置与数据 / 模型上传目录在安装时即建好，
# 避免首次使用时提示目录不存在（服务启动时也会自动确保存在）
echo "  - 创建默认工作目录（runs / datasets / uploads）…"
mkdir -p "$SCRIPT_DIR/runs" "$SCRIPT_DIR/datasets" "$SCRIPT_DIR/uploads"
echo "  - 工作目录就绪"

# ---------- 询问安装配置（端口 / 账号 / 密码） ----------
# 提示用 printf 输出（read -p 在 curl|bash 管道下不显示提示）；
# 优先从 /dev/tty 读取（管道执行时 stdin 不可交互）；60 秒无输入则用默认值；
# 检测不到交互终端时直接使用默认值，保证安装不会中断。
ask_value() {
  local prompt="$1" default="$2"
  printf '%s' "$prompt"
  ASK_VALUE=""
  if [ -t 0 ]; then
    read -r ASK_VALUE || true
  elif [ -e /dev/tty ]; then
    read -r ASK_VALUE < /dev/tty 2>/dev/null || true
  else
    echo "  （检测不到交互终端，使用默认值）"
  fi
  if [ -z "$ASK_VALUE" ]; then ASK_VALUE="$default"; fi
}
ask_secret() {
  local prompt="$1" default="$2"
  printf '%s' "$prompt"
  ASK_VALUE=""
  if [ -t 0 ]; then
    read -r -s ASK_VALUE || true; echo
  elif [ -e /dev/tty ]; then
    read -r -s ASK_VALUE < /dev/tty 2>/dev/null || true; echo
  else
    echo "  （检测不到交互终端，自动生成随机密码）"
  fi
  if [ -z "$ASK_VALUE" ]; then ASK_VALUE="$default"; fi
}
echo "------------------ 安装配置（回车使用默认值） ------------------"
ask_value "  服务端口 [80]: " "80"
SERVER_PORT="$ASK_VALUE"
case "$SERVER_PORT" in
  ''|*[!0-9]*) SERVER_PORT="80"; echo "  端口无效，已使用默认 80" ;;
esac
ask_value "  管理员账号 [admin]: " "admin"
ADMIN_USER="$ASK_VALUE"
ask_secret "  管理员密码（留空自动生成随机密码）: " "$(openssl rand -hex 8 2>/dev/null || head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
ADMIN_PWD="$ASK_VALUE"
echo "  → 端口 ${SERVER_PORT} / 账号 ${ADMIN_USER} / 密码已设置"
if [ "$SERVER_PORT" -lt 1024 ]; then
  echo "  [提示] 端口 ${SERVER_PORT} 低于 1024，服务需以 root 身份运行（建议：sudo visionlab）"
fi
echo "---------------------------------------------------------------"

# ---------- 创建默认管理员账号 ----------
echo "  - 创建默认管理员账号 ${ADMIN_USER} …"
# 用 -c 传代码（heredoc 在 curl|bash 管道下会从 stdin 读取，与管道脚本冲突导致脚本提前退出）
"$VENV_PY" -c '
import sys
sys.path.insert(0, ".")
from pathlib import Path
from server.database import Database
from server.auth import ensure_default_user
db = Database(Path("visionlab.db"))
ensure_default_user(db, sys.argv[1], sys.argv[2])
' "$ADMIN_USER" "$ADMIN_PWD"
echo "  - 管理员账号就绪"
# 保存服务端口，供 start.sh / stop.sh 读取
echo "$SERVER_PORT" > "$SCRIPT_DIR/.visionlab_port"
echo "  - 服务端口 ${SERVER_PORT}（已写入 .visionlab_port）"

# ---------- 6. GPU 驱动检测与自动安装 ----------
# 检测显卡品牌（NVIDIA / AMD / Intel），未安装驱动时自动安装：
#   - NVIDIA：通过 apt 安装官方驱动（Ubuntu 用 ubuntu-drivers 自动选版本）
#   - AMD / Intel：系统内核已内置开源驱动，无需额外安装
# 跳过方式：--no-driver（云主机/虚拟机无独立显卡时自动跳过）
echo "[6/6] GPU 驱动检测…"
GPU_VENDOR=""
if command -v lspci &>/dev/null; then
  if lspci 2>/dev/null | grep -qi "nvidia"; then GPU_VENDOR="nvidia"
  elif lspci 2>/dev/null | grep -qiE "amd|ati|radeon"; then GPU_VENDOR="amd"
  elif lspci 2>/dev/null | grep -qiE "intel|integrated graphics"; then GPU_VENDOR="intel"
  fi
fi

# 按显卡品牌判断驱动是否就绪（NVIDIA 必须 nvidia-smi 可用；
# 不能因为 Intel 集显的 i915 模块存在就误判 NVIDIA 独立显卡已装驱动）
DRIVER_READY=0
if [ -n "$GPU_VENDOR" ]; then
  case "$GPU_VENDOR" in
    nvidia)
      if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then DRIVER_READY=1; fi
      ;;
    amd)
      lsmod 2>/dev/null | grep -q amdgpu && DRIVER_READY=1
      ;;
    intel)
      lsmod 2>/dev/null | grep -q i915 && DRIVER_READY=1
      ;;
  esac
fi

if [ "$DRIVER_READY" = "1" ]; then
  echo "  GPU 驱动: 已就绪"
  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 8 | sed 's/^/    /'
  fi
elif [ -z "$GPU_VENDOR" ]; then
  echo "  GPU: 未检测到独立显卡（可能为云主机/虚拟机，将使用 CPU 训练）"
elif [ "$NO_SUDO" = "1" ]; then
  echo "  检测到 ${GPU_VENDOR} 显卡但未安装驱动；已跳过安装（--no-sudo）"
elif [ "$DRIVER_AUTO" = "0" ]; then
  echo "  检测到 ${GPU_VENDOR} 显卡但未安装驱动；已跳过安装（--no-driver）"
else
  case "$GPU_VENDOR" in
    nvidia)
      echo "  检测到 NVIDIA 显卡但未安装驱动，正在自动安装（可能需要数分钟）…"
      if [ "$(id -u)" -ne 0 ] && ! command -v sudo &>/dev/null; then
        echo "  [错误] 需要 root 或 sudo 权限安装 NVIDIA 驱动，请手动安装后重试"
      else
        SUDO_PREFIX=""
        if [ "$(id -u)" -ne 0 ]; then SUDO_PREFIX="sudo"; fi
        DRV_OK=0
        if command -v ubuntu-drivers &>/dev/null && $SUDO_PREFIX ubuntu-drivers install; then
          DRV_OK=1
        fi
        if [ "$DRV_OK" = "0" ]; then
          $SUDO_PREFIX apt-get update -y
          if $SUDO_PREFIX apt-get install -y nvidia-driver-535; then
            DRV_OK=1
          elif $SUDO_PREFIX apt-get install -y nvidia-driver-470; then
            DRV_OK=1
          fi
        fi
        if [ "$DRV_OK" = "1" ]; then
          echo "  NVIDIA 驱动安装完成"
          echo "  [重要] 重启系统后驱动生效：sudo reboot"
          echo "         重启前服务将使用 CPU 训练，重启后自动启用 GPU。"
        else
          echo "  [警告] NVIDIA 驱动自动安装失败，请手动执行："
          echo "         sudo ubuntu-drivers install"
          echo "         （需驱动版本 ≥ 525 以支持 CUDA 12.1，如 nvidia-driver-535）"
        fi
      fi
      ;;
    amd|intel)
      echo "  显卡: ${GPU_VENDOR}（系统内核已内置开源驱动，无需额外安装）"
      ;;
  esac
fi

echo ""
echo "=========================================="
echo " VisionLab 安装完成！"
echo "=========================================="
SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -z "$SERVER_IP" ]; then SERVER_IP="localhost"; fi
echo "  页面地址:   http://${SERVER_IP}:${SERVER_PORT}"
echo "  本机访问:   http://127.0.0.1:${SERVER_PORT}"
echo ""
echo "  登录账号:   ${ADMIN_USER}"
echo "  登录密码:   ${ADMIN_PWD}"
echo "  （请登录后立即在「设置」页修改密码）"
echo ""
echo "  仓库目录:   $SCRIPT_DIR"
echo ""
echo "  启动服务:   visionlab"
echo "  停止服务:   visionlab-stop"
echo "  删除程序:   visionlab-uninstall"
echo "  开发模式:   visionlab --dev"
if [ "$SERVER_PORT" -lt 1024 ] && [ "$(id -u)" -ne 0 ]; then
  echo "  [提示] 端口 ${SERVER_PORT} 低于 1024，启动/停止请用 sudo 执行"
fi
echo ""
echo "  训练输出:   $SCRIPT_DIR/runs"
echo "  数据集:     $SCRIPT_DIR/datasets"
echo "  训练环境:   YOLOv11 / YOLOv8 / YOLOv5（Conda，Python 3.11）"
echo "=========================================="

# ---------- 配置开机自启（systemd） ----------
# 服务器重启后自动启动服务；不使用 systemd 的环境可跳过（失败不影响当前启动）
SERVICE_NAME="visionlab"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SVC_OK=0
write_service_unit() {
  printf '%s\n' \
    "[Unit]" \
    "Description=VisionLab YOLO Training Manager" \
    "After=network-online.target" \
    "Wants=network-online.target" \
    "" \
    "[Service]" \
    "Type=simple" \
    "WorkingDirectory=$SCRIPT_DIR" \
    "ExecStart=/bin/bash $SCRIPT_DIR/start.sh" \
    "Restart=on-failure" \
    "RestartSec=5" \
    "" \
    "[Install]" \
    "WantedBy=multi-user.target"
}
if [ "$(id -u)" -eq 0 ]; then
  if write_service_unit > "$SERVICE_FILE" 2>/dev/null \
     && systemctl daemon-reload 2>/dev/null \
     && systemctl enable "$SERVICE_NAME" >/dev/null 2>&1; then
    SVC_OK=1
  fi
elif command -v sudo &>/dev/null; then
  if write_service_unit | sudo tee "$SERVICE_FILE" >/dev/null 2>&1 \
     && sudo systemctl daemon-reload 2>/dev/null \
     && sudo systemctl enable "$SERVICE_NAME" >/dev/null 2>&1; then
    SVC_OK=1
  fi
fi
if [ "$SVC_OK" = "1" ]; then
  echo "  开机自启: 已启用（systemd: ${SERVICE_NAME}.service）"
  echo "  重启后服务将自动启动；当前会话仍立即启动（见下）。"
else
  echo "  [提示] 未配置开机自启（无 systemd 或权限不足）。"
  echo "         如需重启自动启动，请手动执行："
  echo "         sudo systemctl enable visionlab"
fi

# ---------- 创建全局启动命令（任意目录可用） ----------
# 解决用户在非仓库目录执行 ./start.sh 时报"没有那个文件或目录"的问题
# start/stop/uninstall 脚本内部用 readlink -f 解析真实仓库路径，软链接执行不影响
if [ "$(id -u)" -eq 0 ]; then
  ln -sf "$SCRIPT_DIR/start.sh" /usr/local/bin/visionlab 2>/dev/null \
    && echo "  全局命令: visionlab / visionlab-stop / visionlab-uninstall（任意目录可用）"
  ln -sf "$SCRIPT_DIR/stop.sh" /usr/local/bin/visionlab-stop 2>/dev/null
  ln -sf "$SCRIPT_DIR/uninstall.sh" /usr/local/bin/visionlab-uninstall 2>/dev/null
fi

START_LOG="$SCRIPT_DIR/visionlab-start.log"
echo "  正在自动启动服务…"
nohup bash "$SCRIPT_DIR/start.sh" > "$START_LOG" 2>&1 &
START_OK=0
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${SERVER_PORT}/api/health" >/dev/null 2>&1; then
    echo "  服务已启动 ✓  页面地址: http://${SERVER_IP}:${SERVER_PORT}"
    START_OK=1
    break
  fi
  sleep 1
done
if [ "$START_OK" = "0" ]; then
  echo "  [警告] 服务未就绪，最近日志："
  tail -n 15 "$START_LOG" 2>/dev/null | sed 's/^/    /' || true
  echo "  [提示] 启动命令: visionlab"
  if grep -q "bind on address" "$START_LOG" 2>/dev/null; then
    echo "  [提示] 端口 ${SERVER_PORT} 绑定被拒绝（环境权限限制），请改用高位端口，例如："
    echo "         echo 8080 > \"$SCRIPT_DIR/.visionlab_port\" && visionlab"
  elif [ "$SERVER_PORT" -lt 1024 ] && [ "$(id -u)" -ne 0 ]; then
    echo "  [提示] 端口 ${SERVER_PORT} 低于 1024，普通用户无法直接绑定，请用："
    echo "         sudo visionlab"
  fi
fi
echo "  启动日志:   $START_LOG"
echo "=========================================="

# ---------- 安装完成：是否需要重启（最后一个交互询问，在服务启动之后） ----------
# 行为明确可预期：仅本次安装了 NVIDIA 驱动才需要重启（重启后驱动加载、systemd 自启生效）；
# 未装驱动则无需重启。先展示服务启动结果，最后再询问是否重启。
if [ "$DRV_OK" = "1" ]; then
  echo ""
  ask_value " 是否现在重启服务器？[Y/n]（回车默认重启） " "Y"
  case "$ASK_VALUE" in
    y|Y|yes|YES)
      echo " 正在重启服务器…（重启后 systemd 将自动启动服务）"
      reboot
      exit 0
      ;;
    *)
      echo " 已选择稍后重启，服务保持运行（当前以 CPU 模式训练）。"
      echo " 需要 GPU 训练时请手动执行：sudo reboot"
      ;;
  esac
fi
echo "=========================================="
