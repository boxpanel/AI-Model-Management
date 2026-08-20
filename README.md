# VisionLab · AI Model Management

YOLO 训练管理平台（AI Model Management）：基于 Web 的可视化 YOLO 训练、数据集与模型管理系统，开箱即用，支持多 GPU 训练、模型格式转换（含瑞芯微 RKNN）、训练不中断与多用户实时监控。

## ✨ 功能特性

- **训练管理**
  - 可视化配置 YOLOv11 / YOLOv8 / YOLOv5 训练参数（轮次、batch、imgsz、学习率、优化器等）
  - 多 GPU 勾选训练（自动检测全部显卡，生成 `device="0,1"`）
  - 实时监控：loss / mAP50 曲线、进度、ETA、日志（WebSocket 推送）
  - 训练不中断：独立子进程 + 状态持久化，Web 服务重启 / 页面关闭训练继续运行，重开页面自动恢复监控
- **数据集**
  - 内置 coco8 / coco128 预设，支持新建 / 双击编辑 / 删除
  - 上传 `.zip` 数据集压缩包自动解压
  - 完整的 YAML 字段管理（path / train / val / test / nc / names / download）
- **模型仓库**
  - 权重上传（.pt / .onnx）、下载、删除
  - 训练产出（best.pt / last.pt）自动入库，显示最后训练时间
  - **模型格式转换**：ONNX、TensorRT、TFLite、TorchScript、OpenVINO、NCNN、**RKNN（瑞芯微 NPU）**，后台子进程执行，可取消
- **任务与草稿**
  - 训练任务历史持久化（SQLite），支持详情查看与删除
  - 草稿保存 / 一键恢复
- **环境与硬件**
  - 一键检测 Python / Ultralytics / CUDA / GPU 环境
  - 实时硬件监控（CPU / 内存 / 多 GPU 使用率）

## 📁 目录结构

```
.
├── index.html          # 单页前端（无框架，离线降级模式）
├── install.sh          # 一键安装脚本（系统依赖 + Python 依赖 + RKNN 工具链）
├── start.sh            # 启动脚本（./start.sh 生产 / ./start.sh --dev 开发）
├── requirements.txt    # Python 依赖
└── server/             # FastAPI 后端
    ├── main.py         # API 与 WebSocket
    ├── training.py     # 训练管理器（子进程 + 状态轮询）
    ├── train_worker.py # 训练子进程（独立运行，不中断）
    ├── convert_worker.py  # 模型转换子进程
    ├── models_convert.py  # 转换任务管理器（可取消）
    ├── models_repo.py  # 模型仓库扫描
    ├── datasets.py     # 数据集服务
    ├── database.py     # SQLite 持久化
    ├── hardware.py     # CPU / 内存 / GPU 监控
    └── environment.py  # 环境检测
```

## 🚀 一键安装

要求：**Linux（Ubuntu/Debian 或 CentOS/RHEL）**，Python 3.10-3.12，root/sudo 权限。

在服务器上执行**一条命令**即可完成下载源码 + 安装全部依赖 + 启动服务：

```bash
curl -fsSL https://raw.githubusercontent.com/boxpanel/AI-Model-Management/main/install.sh | bash && ./start.sh
```

该命令自动处理：首次安装会克隆仓库到 `~/AI-Model-Management`；若目录已存在则自动 `git pull` 更新（**不会因目录已存在而报错**）。

只想安装不启动，或已启动过需要重装：

```bash
curl -fsSL https://raw.githubusercontent.com/boxpanel/AI-Model-Management/main/install.sh | bash
```

手动方式（已在服务器上克隆过仓库）：

```bash
cd AI-Model-Management && git pull && bash install.sh && ./start.sh
```

`install.sh` 会自动完成：

1. 检测 Python 版本
2. 安装系统依赖（apt/dnf：python3-dev、libxslt1-dev、libgl1、protobuf、gcc 等）
3. 安装 Python 依赖（fastapi / uvicorn / ultralytics / torch 等）
4. 安装 rknn-toolkit2（RKNN 转换需要，失败不影响其他功能）
5. 检测 NVIDIA 驱动 / CUDA（GPU 训练可用性）

参数：`--skip-rknn` 跳过 RKNN 工具链；`--no-sudo` 跳过系统依赖安装。

## ▶️ 启动服务

```bash
bash install.sh && ./start.sh          # 安装并启动（生产模式）
./start.sh --dev                       # 开发模式（热重载）
```

启动后浏览器访问 `http://<服务器IP>:8000` 即可使用。生产模式下训练不受热重载影响。

## 🧠 模型转换支持格式

| 格式 | 说明 | 产物 |
|---|---|---|
| ONNX | 跨框架通用 | `.onnx` |
| TensorRT | NVIDIA GPU 加速（需 GPU） | `.engine` |
| TFLite | 移动端 / 边缘设备 | `.tflite` |
| TorchScript | PyTorch 原生 | `.torchscript` |
| OpenVINO | Intel CPU 加速 | 目录 |
| NCNN | 移动端推理 | 目录 |
| RKNN | 瑞芯微 Rockchip NPU（需 rknn-toolkit2） | `.rknn` |

转换在后台子进程执行，支持取消，产物自动进入模型仓库。

## 🔧 常见问题

- **GPU 训练**：需先安装 NVIDIA 驱动（`nvidia-smi` 可见），`install.sh` 会检测并提示。
- **RKNN 转换失败**：确认已安装 `rknn-toolkit2`（仅 Linux x86_64/aarch64 + Python 3.8-3.12），并指定正确的目标平台（默认 rk3588）。
- **自定义数据集**：先在"数据集"页上传压缩包或新建 yaml，再到训练页选择。

## 🛠 技术栈

FastAPI · WebSocket · SQLite · Ultralytics YOLO · 原生 HTML/JS（无构建步骤）

## 📄 License

MIT
