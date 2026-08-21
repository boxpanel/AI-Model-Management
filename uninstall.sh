#!/usr/bin/env bash
# VisionLab 删除程序
# 停止服务并删除：虚拟环境、数据库、端口配置、训练输出与数据集
# 注意：Conda 训练环境（YOLOv11 / YOLOv8 / YOLOv5）不会自动删除，删除方式见下方提示
set -e
# readlink -f 解析软链接真实路径，保证通过全局命令 visionlab-uninstall 执行时 SCRIPT_DIR 指向仓库目录
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

echo "=========================================="
echo " VisionLab 删除程序"
echo "=========================================="
echo "将删除以下内容："
echo "  - 服务进程（如正在运行）"
echo "  - venv 虚拟环境"
echo "  - visionlab.db 数据库"
echo "  - .visionlab_port 端口配置"
echo "  - runs/ 训练输出"
echo "  - datasets/ 数据集"
echo "=========================================="

ANS=""
if [ -t 0 ]; then
  read -r -p "确定继续删除？[y/N] " ANS || true
else
  read -r -p "确定继续删除？[y/N] " ANS < /dev/tty 2>/dev/null || true
fi
case "$ANS" in
  y|Y|yes|YES) ;;
  *) echo "已取消删除。"; exit 0 ;;
esac

echo "[1/3] 停止服务…"
bash "$SCRIPT_DIR/stop.sh" || true
sleep 1

# 移除开机自启服务（如已注册）
UNINSTALL_SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &>/dev/null; then UNINSTALL_SUDO="sudo"; fi
if [ -f "/etc/systemd/system/visionlab.service" ]; then
  echo "  - 移除开机自启服务 visionlab …"
  $UNINSTALL_SUDO systemctl stop visionlab 2>/dev/null || true
  $UNINSTALL_SUDO systemctl disable visionlab 2>/dev/null || true
  $UNINSTALL_SUDO rm -f /etc/systemd/system/visionlab.service
  $UNINSTALL_SUDO systemctl daemon-reload 2>/dev/null || true
fi
# 移除全局启动命令软链接
$UNINSTALL_SUDO rm -f /usr/local/bin/visionlab /usr/local/bin/visionlab-stop /usr/local/bin/visionlab-uninstall 2>/dev/null || true

echo "[2/3] 删除程序文件…"
rm -rf "$SCRIPT_DIR/venv"
rm -f "$SCRIPT_DIR/visionlab.db" "$SCRIPT_DIR/.visionlab_port"
rm -rf "$SCRIPT_DIR/runs" "$SCRIPT_DIR/datasets"
find "$SCRIPT_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "[3/3] 清理完成。"
echo ""
echo " 仓库代码目录保留在：$SCRIPT_DIR"
echo " 如需彻底删除整个程序目录，请执行："
echo "   cd ~ && rm -rf '$SCRIPT_DIR'"
echo ""
echo " 如需删除 Conda 训练环境（可选）："
echo "   conda env remove -n YOLOv11 -y"
echo "   conda env remove -n YOLOv8 -y"
echo "   conda env remove -n YOLOv5 -y"
