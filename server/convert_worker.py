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
import threading
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


def _onnx_matches_imgsz(onnx_path: Path, imgsz: int) -> bool:
    """判断已有 ONNX 文件的输入尺寸是否与目标 imgsz 一致（不一致时不能复用）。

    YOLO 导出 ONNX 输入为 [1,3,H,W]（或 [1,3,W,H]），末两维之一等于 imgsz 即匹配。
    """
    if not onnx_path.is_file():
        return False
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        for inp in model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if imgsz in dims:
                return True
        return False
    except Exception:
        return False


class WorkerState:
    """state 文件读写（临时文件 + rename 原子写，避免读到半截内容）。

    同进程内多线程（进度推进线程与主线程）并发 update 存在读-改-写竞态，
    可能覆盖彼此刚写入的字段，故用线程锁串行化。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update(self, **fields: Any) -> None:
        with self._lock:
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

    # 自定义输出目录与名称（默认：源文件所在目录 + 源文件名）
    out_dir = Path((config.get("output_dir") or "").strip() or str(src_path.parent))
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = (config.get("name") or "").strip() or src_path.stem
    # 导出产物默认落在源文件目录并以源文件名命名；仅在目标位置与默认不一致时才需要移动/重命名
    need_relocate = str(out_dir) != str(src_path.parent.resolve()) or out_name != src_path.stem

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
            # 第一步：导出 ONNX（rknn-toolkit2 从 ONNX 转换，opset 12 兼容性最佳）。
            # 若输出目录已存在相同输入尺寸的 ONNX 文件则直接复用，避免重复导出耗时
            imgsz = config.get("imgsz", 640)
            onnx_candidate = out_dir / (out_name + ".onnx")
            if _onnx_matches_imgsz(onnx_candidate, imgsz):
                state.update(state="running", message="正在加载已有 ONNX 中间文件…", progress=15)
                onnx_out = onnx_candidate
            else:
                state.update(state="running", message="正在导出 ONNX 中间文件…", progress=10)
                onnx_out = Path(
                    str(YOLO(str(src_path)).export(format="onnx", imgsz=imgsz, opset=12, dynamic=False, verbose=False))
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
            out = out_dir / (out_name + ".rknn")
            rknn = RKNN(verbose=False)
            try:
                rknn.config(
                    mean_values=[[0, 0, 0]],
                    std_values=[[255, 255, 255]],
                    target_platform=platform,
                    # rknn-toolkit2 2.3.2 的 fuse_mul_into_sdpa 融合规则对含 SDPA 注意力
                    # 模块的模型（如 YOLOv11）有 bug，构建时报
                    # TypeError: only 0-dimensional arrays can be converted to Python scalars，
                    # 官方提示通过 disable_rules 禁用该规则
                    disable_rules=["fuse_mul_into_sdpa"],
                )
                if rknn.load_onnx(model=str(onnx_out)) != 0:
                    raise RuntimeError("加载 ONNX 模型失败")
                # 构建 IR 图在 CPU 上进行，首次构建可能需要数分钟，先更新状态避免用户误以为卡死
                state.update(state="running", message="正在构建 RKNN 模型（首次构建可能需要数分钟，请耐心等待）…", progress=45)
                # 构建期间用后台线程平滑推进进度（45 → 80），让用户直观看到仍在进行。
                # 注意：pump 与主线程并发写 state 文件存在竞态（可能覆盖 completed），
                # 故主线程用 done 事件通知 pump 立即退出，确保 completed 为最后一次写入。
                done = threading.Event()

                def _progress_pump() -> None:
                    p = 45
                    while p < 80 and not done.is_set() and not stop_event["stop"]:
                        if done.wait(5):  # 主线程已完成，立即退出不再写 state
                            return
                        p += 1
                        state.update(progress=p)

                pump = threading.Thread(target=_progress_pump, daemon=True)
                pump.start()
                try:
                    if rknn.build(do_quantization=False) != 0:
                        raise RuntimeError("构建 RKNN 模型失败")
                    state.update(state="running", message="正在导出 RKNN 文件…", progress=85)
                    import shutil
                    # rknn-toolkit2 部分版本会把「输出路径」当作目录写入 <名称>/best.rknn，
                    # 预创建该目录避免 FileNotFoundError；导出后把产物归位到目标文件
                    alt_dir = out.parent / (out.name[:-5] if out.name.lower().endswith(".rknn") else out.name)
                    alt_dir.mkdir(parents=True, exist_ok=True)
                    if rknn.export_rknn(str(out)) != 0:
                        raise RuntimeError("导出 RKNN 模型失败")
                    if not (out.exists() and out.is_file()):
                        # 目标文件未生成：递归查找输出目录下的 .rknn 产物并归位到目标文件。
                        # 个别版本会把 out 本身建成目录（<out>/best.rknn），移动前先清理该目录
                        for c in sorted(out.parent.rglob("*.rknn")):
                            if c != out and c.is_file():
                                if out.exists():
                                    if out.is_dir():
                                        shutil.rmtree(out, ignore_errors=True)
                                    else:
                                        out.unlink(missing_ok=True)
                                c.replace(out)
                                break
                    if alt_dir.exists() and alt_dir.is_dir():
                        shutil.rmtree(alt_dir, ignore_errors=True)
                    state.update(progress=95)
                finally:
                    done.set()
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
            # NCNN 首次导出需下载 pnnx 工具链、OpenVINO 转换耗时，过程无反馈容易误以为卡死。
            # 提示可能耗时，并用后台线程平滑推进进度（15 → 85）。
            # pump 与主线程并发写 state 存在竞态（可能覆盖 completed），故用 done 事件通知退出。
            hint = "（首次转换 NCNN 需下载 pnnx 工具链，可能需要数分钟，请耐心等待）" if fmt == "ncnn" else ""
            state.update(state="running", message=f"正在导出为 {fmt}…{hint}", progress=15)
            done = threading.Event()

            def _progress_pump() -> None:
                p = 15
                while p < 85 and not done.is_set() and not stop_event["stop"]:
                    if done.wait(5):  # 导出已完成，立即退出不再写 state
                        return
                    p += 1
                    state.update(progress=p)

            pump = threading.Thread(target=_progress_pump, daemon=True)
            pump.start()
            try:
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
            finally:
                done.set()
            state.update(progress=95)
            # 自定义输出目录/名称：导出产物默认在源文件目录（以源文件名命名），
            # 目标位置不同或名称不同时移动/重命名到目标位置（目录型保留 _openvino_model 等后缀）
            if need_relocate:
                import shutil
                if out_path.is_dir():
                    dest = out_dir / (out_name + out_path.name[len(src_path.stem):])
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(out_path), str(dest))
                else:
                    dest = out_dir / (out_name + out_path.suffix)
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(out_path), str(dest))
                out_path = dest
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
