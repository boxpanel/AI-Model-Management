"""模型转换子进程：独立于 Web 服务运行，将 YOLO 权重导出为其他推理格式。

设计目的：
- 转换在独立进程中执行，不阻塞 Web 服务；
- 状态实时原子写入 state 文件，服务端查询该文件即可获取进度；
- 产物默认导出到源文件同目录，自动进入模型仓库列表；
- 支持 SIGTERM/SIGINT 优雅停止。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ultralytics export() 支持的格式 -> 产物形态（文件或目录后缀）
# rknn 为瑞芯微 NPU 专用格式，由 rknn-toolkit2 转换（先导出 ONNX 再转换）
FORMAT_INFO = {
    "onnx": {"ext": ".onnx", "dir": False},
    "engine": {"ext": ".engine", "dir": False},
    "torchscript": {"ext": ".torchscript", "dir": False},
    "openvino": {"ext": "_openvino_model", "dir": True},
    "ncnn": {"ext": "_ncnn_model", "dir": True},
    "rknn": {"ext": ".rknn", "dir": False},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkerState:
    """state 文件读写（临时文件 + rename 原子写，避免读到半截内容）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update(self, **fields: Any) -> None:
        data = self.read()
        data.update(fields)
        data["updated_at"] = _utc_now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 模型转换子进程")
    parser.add_argument("--config", required=True, help="转换配置 JSON 文件路径")
    parser.add_argument("--state", required=True, help="状态文件输出路径")
    args = parser.parse_args()

    state = WorkerState(Path(args.state))
    state.update(pid=os.getpid())
    stop_event = {"stop": False}

    def handle_term(signum: int, frame: Any) -> None:
        stop_event["stop"] = True
        state.update(state="stopped", message="转换已取消")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except Exception as exc:
        state.update(state="error", message=f"读取配置失败：{exc}")
        return 1

    base_dir = Path(config.get("base_dir", ".")).resolve()
    # 注意：ConvertStartRequest 的字段名为 path（与前端一致），source 仅为历史兼容
    source = config.get("path") or config.get("source", "")
    fmt = str(config.get("fmt", "onnx")).lower()
    if fmt not in FORMAT_INFO:
        state.update(state="error", message=f"不支持的导出格式：{fmt}")
        return 1

    # 确保 pkg_resources（setuptools）可用：ultralytics / rknn-toolkit2 内部依赖它，缺失即失败。
    # 注意：setuptools 82+ 已移除 pkg_resources，自动补装必须固定兼容版本（<=80.10.2）。
    def _ensure_pkg_resources() -> bool:
        try:
            import pkg_resources  # noqa: F401
            return True
        except ImportError:
            pass
        state.update(state="running", message="检测到缺少 pkg_resources，正在自动安装 setuptools…")
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "setuptools<=80.10.2"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except Exception:
            pass
        try:
            import pkg_resources  # noqa: F401
            return True
        except ImportError:
            return False

    _ensure_pkg_resources()

    def _ensure_onnx_compat() -> bool:
        """确保 onnx 版本与 rknn-toolkit2 兼容（rknn-toolkit2 依赖顶层 onnx.mapping，新版 onnx 已移除）。

        新版 onnx（1.19+）移除了顶层 mapping 属性，rknn-toolkit2 旧代码在 load_onnx 时会抛出
        AttributeError: module 'onnx' has no attribute 'mapping'。此处检测到不兼容时自动安装
        兼容版本 onnx>=1.16.1,<1.19.0（满足 rknn-toolkit2>=1.16.1 要求且保留 mapping），
        并用 os.execv 重启自身进程（PID 不变，服务端 Popen 句柄
        与 state 文件轮询均不受影响），避免用户手动装错解释器。
        """
        import subprocess

        try:
            import onnx
        except ImportError:
            compatible, version = False, "未安装"
        else:
            compatible = hasattr(onnx, "mapping")
            version = getattr(onnx, "__version__", "未知")
        if compatible:
            return True
        if os.environ.get("VISIONLAB_ONNX_FIXED") == "1":
            # 已自动安装过一次仍不兼容：说明装到了错误解释器，避免无限重启循环，直接给出指引
            state.update(
                state="error",
                message=(
                    f"onnx {version} 与 rknn-toolkit2 不兼容（onnx 1.19+ 移除了 mapping），"
                    "自动安装 onnx>=1.16.1,<1.19.0 后仍未生效，请手动执行："
                    "python -m pip install \"onnx>=1.16.1,<1.19.0\""
                ),
            )
            return False
        state.update(
            state="running",
            message=f"检测到 onnx {version} 与 rknn-toolkit2 不兼容，正在自动安装 onnx>=1.16.1,<1.19.0…",
            progress=8,
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "onnx>=1.16.1,<1.19.0"],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except Exception as exc:
            state.update(state="error", message=f"自动安装 onnx 兼容版本失败：{exc}")
            return False
        if result.returncode != 0:
            state.update(
                state="error",
                message=f"自动安装 onnx 兼容版本失败：{result.stderr.strip()[-300:]}",
            )
            return False
        state.update(state="running", message="onnx 兼容版本安装成功，正在重启转换进程使新版本生效…", progress=9)
        os.environ["VISIONLAB_ONNX_FIXED"] = "1"
        # execv 用同一 PID 替换当前进程映像（仅 Linux/Unix 支持，本项目部署于 Ubuntu 服务器）
        os.execv(sys.executable, [sys.executable] + sys.argv)
        return True  # 不可达，仅满足类型检查

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        if "pkg_resources" in str(exc) or "setuptools" in str(exc):
            state.update(state="error", message="缺少 pkg_resources（setuptools），请执行：python -m pip install \"setuptools<=80.10.2\"")
        else:
            state.update(state="error", message="未安装 ultralytics，请先执行 pip install -r requirements.txt")
        return 1

    src_path = Path(source)
    if not src_path.is_absolute():
        src_path = (base_dir / src_path).resolve()
    if not src_path.exists() or not src_path.is_file():
        state.update(state="error", message=f"源文件不存在：{source}")
        return 1

    device = config.get("device", "cpu")
    # TensorRT 导出必须在 GPU 上进行，自动切换到 0 号卡
    if fmt == "engine" and str(device).lower() == "cpu":
        device = "0"

    state.update(state="running", message=f"正在导出为 {fmt}…", progress=5)

    try:
        if fmt == "rknn":
            # 第零步：确保 onnx 版本与 rknn-toolkit2 兼容（不兼容时自动修复并重启自身进程）
            if not _ensure_onnx_compat():
                return 1
            # 第一步：导出 ONNX（rknn-toolkit2 从 ONNX 转换，opset 12 兼容性最佳）
            state.update(state="running", message="正在导出 ONNX 中间文件…", progress=10)
            onnx_out = Path(
                str(YOLO(str(src_path)).export(format="onnx", imgsz=config.get("imgsz", 640), opset=12, dynamic=False, verbose=False))
            )
            state.update(state="running", message="正在加载 ONNX 模型…", progress=30)
            # 第二步：rknn-toolkit2 将 ONNX 转换为 RKNN（内部依赖 pkg_resources，先确保可用）
            if not _ensure_pkg_resources():
                state.update(state="error", message="缺少 pkg_resources（setuptools）且自动安装失败，请执行：python -m pip install \"setuptools<=80.10.2\"")
                return 1
            try:
                from rknn.api import RKNN
            except ImportError as exc:
                if "pkg_resources" in str(exc):
                    state.update(state="error", message="rknn-toolkit2 依赖 pkg_resources，请执行：python -m pip install \"setuptools<=80.10.2\"")
                else:
                    state.update(state="error", message="未安装 rknn-toolkit2，无法转换 RKNN 格式，请先 pip install rknn-toolkit2")
                return 1
            platform = str(config.get("rknn_platform", "rk3588"))
            out = src_path.with_suffix(".rknn")
            rknn = RKNN(verbose=False)
            try:
                rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
                if rknn.load_onnx(model=str(onnx_out)) != 0:
                    raise RuntimeError("加载 ONNX 模型失败")
                # 构建 IR 图在 CPU 上进行，首次构建可能需要数分钟，先更新状态避免用户误以为卡死
                state.update(state="running", message="正在构建 RKNN 模型（首次构建可能需要数分钟，请耐心等待）…", progress=45)
                # 构建期间用后台线程平滑推进进度（45 → 80），让用户直观看到仍在进行
                import threading
                def _progress_pump() -> None:
                    import time as _t
                    step = 0
                    while step < 35 and not stop_event["stop"]:  # 35 × 5s ≈ 3 分钟封顶
                        _t.sleep(5)
                        step += 1
                        cur = state.read().get("progress", 0) or 0
                        if cur >= 80:
                            break
                        state.update(progress=min(80, cur + 1))
                pump = threading.Thread(target=_progress_pump, daemon=True)
                pump.start()
                if rknn.build(do_quantization=False) != 0:
                    raise RuntimeError("构建 RKNN 模型失败")
                state.update(state="running", message="正在导出 RKNN 文件…", progress=85)
                if rknn.export_rknn(str(out)) != 0:
                    raise RuntimeError("导出 RKNN 模型失败")
                state.update(progress=95)
            except AttributeError as exc:
                if "onnx" in str(exc):
                    # onnx 1.19+ 移除了顶层 mapping 属性，rknn-toolkit2 旧代码依赖它
                    raise RuntimeError(
                        "onnx 版本与 rknn-toolkit2 不兼容（onnx 1.19+ 移除了 mapping），"
                        "请执行：python -m pip install \"onnx>=1.16.1,<1.19.0\""
                    ) from exc
                raise
            finally:
                rknn.release()
            out_path = out
        else:
            state.update(state="running", message=f"正在导出为 {fmt}…", progress=15)
            out_path = Path(
                str(
                    YOLO(str(src_path)).export(
                        format=fmt,
                        imgsz=config.get("imgsz", 640),
                        device=device,
                        verbose=False,
                    )
                )
            )
            state.update(progress=95)
        if stop_event["stop"]:
            state.update(state="stopped", message="转换已停止")
            return 0
        out = Path(str(out_path))
        try:
            rel = str(out.relative_to(base_dir)).replace("\\", "/")
        except ValueError:
            rel = str(out)
        state.update(
            state="completed",
            progress=100.0,
            output=rel,
            message="转换完成",
        )
    except Exception as exc:
        detail = traceback.format_exc()
        state.update(state="error", message=f"转换失败：{exc}", detail=detail[-2000:])
        print(detail, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
