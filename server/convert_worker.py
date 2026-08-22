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
    "tflite": {"ext": ".tflite", "dir": False},
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

    try:
        from ultralytics import YOLO
    except ImportError:
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

    state.update(state="running", message=f"正在导出为 {fmt}…")

    try:
        if fmt == "rknn":
            # 第一步：导出 ONNX（rknn-toolkit2 从 ONNX 转换，opset 12 兼容性最佳）
            onnx_out = Path(
                str(YOLO(str(src_path)).export(format="onnx", imgsz=config.get("imgsz", 640), opset=12, dynamic=False, verbose=False))
            )
            # 第二步：rknn-toolkit2 将 ONNX 转换为 RKNN
            try:
                from rknn.api import RKNN
            except ImportError:
                state.update(state="error", message="未安装 rknn-toolkit2，无法转换 RKNN 格式，请先 pip install rknn-toolkit2")
                return 1
            platform = str(config.get("rknn_platform", "rk3588"))
            out = src_path.with_suffix(".rknn")
            rknn = RKNN(verbose=False)
            try:
                rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
                if rknn.load_onnx(model=str(onnx_out)) != 0:
                    raise RuntimeError("加载 ONNX 模型失败")
                if rknn.build(do_quantization=False) != 0:
                    raise RuntimeError("构建 RKNN 模型失败")
                if rknn.export_rknn(str(out)) != 0:
                    raise RuntimeError("导出 RKNN 模型失败")
            finally:
                rknn.release()
            out_path = out
        else:
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
