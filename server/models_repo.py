from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# 模型仓库可识别的单文件扩展名（含转换产物）
FILE_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".xml", ".rknn"}
# 目录型转换产物（ultralytics 导出为目录：openvino / ncnn / tfjs / saved_model / paddle）
DIR_MARKER = "_model"


def _guess_model_version(name: str) -> str:
    """从权重文件名解析模型版本（yolo11n.pt → YOLO11n、yolov5nu.pt → YOLOv5nu、yolo11x.onnx → YOLO11x、yolo11n.yaml → YOLO11n），无法识别返回空。"""
    m = re.match(r"^yolo(11|v8|v5)([nsmlx])(u)?\.(pt|onnx|engine|torchscript|xml|rknn|yaml)$", name.lower())
    if not m:
        return ""
    fam, size, u = m.groups()
    prefix = {"11": "YOLO11", "v8": "YOLOv8", "v5": "YOLOv5"}[fam]
    return prefix + size + (u or "")


def _mode_from_weights(wp: str) -> str:
    """从训练配置的权重路径判断训练方式：.yaml = 从0训练（scratch），.pt = 预训练/续训（pretrained）。"""
    low = (wp or "").strip().lower()
    if low.endswith(".yaml"):
        return "scratch"
    if low.endswith(".pt"):
        return "pretrained"
    return ""


def _model_version_for(
    path: Path, source: str, task_versions: dict[Any, str], task_name: str | None = None
) -> str:
    """解析权重条目的模型版本：文件名 → 同目录源 .pt（转换产物继承）→ 训练任务记录（按大类+任务名）→ 输出目录中间层大类。"""
    ver = _guess_model_version(path.name)
    if ver:
        return ver
    # 转换产物（best.onnx 等，与源 .pt 同名换扩展名）：继承同目录源 .pt 的版本
    src_pt = path.with_suffix(".pt")
    if path.is_file() and src_pt != path and src_pt.exists():
        ver = _guess_model_version(src_pt.name)
        if ver:
            return ver
    if source == "training":
        tn = task_name or path.parent.parent.name
        # 输出目录中间层大类（runs/YOLO11/<任务>/weights/... → YOLO11）。
        # 同名任务在不同大类文件夹下会各有一条训练记录，必须按「大类+任务名」取版本，
        # 否则最新一次训练（如 YOLOv5lu）会错误覆盖其他大类产物显示的版本。
        parts = path.parts
        family = parts[parts.index("runs") + 1] if "runs" in parts else ""
        ver = task_versions.get((family, tn), "") if family else ""
        if not ver:
            ver = task_versions.get(tn, "")  # 旧记录无大类信息时的兜底
        if not ver:
            ver = family  # 最终兜底：大类
        return ver
    return ""


def _family_of_task(task: dict[str, Any]) -> str:
    """从任务记录的输出目录（runs/<大类>/<任务>）或 model_version 推导模型大类，无法识别返回空。"""
    out = (task.get("output_dir") or "").strip().replace("\\", "/")
    if out:
        parts = [p for p in out.split("/") if p]
        if "runs" in parts:
            idx = parts.index("runs")
            if idx + 1 < len(parts):
                fam = parts[idx + 1]
                if fam != "tasks":
                    return fam
    return str(task.get("model_version") or "").strip()


def _entry_size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def _entry_info(base_dir: Path, path: Path, source: str, name: str) -> dict[str, Any]:
    rel = str(path.relative_to(base_dir)).replace("\\", "/")
    try:
        size_mb = round(_entry_size(path) / (1024 * 1024), 2)
        mtime = path.stat().st_mtime
    except OSError:
        size_mb, mtime = 0.0, 0.0
    return {
        "name": name,
        "path": rel,
        "source": source,
        "size_mb": size_mb,
        "updated_at": datetime.fromtimestamp(mtime).isoformat(),
    }


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and DIR_MARKER in path.name


def list_models(base_dir: Path, runs_dirs: list[Path] | None = None, db=None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    uploads = base_dir / "uploads"
    runs_list = runs_dirs or [base_dir / "runs"]

    # 任务名 → 模型版本 映射（训练产出 best.pt 从任务记录查询实际使用的模型版本）
    # 键为 (大类, 任务名)：同名任务在 runs/YOLO11、YOLOv8、YOLOv5 下各有一次训练记录，
    # 仅按任务名建键会让最新一次训练的错误覆盖所有大类的产物显示
    task_versions: dict[Any, str] = {}
    # 任务名 → 训练方式（scratch 从0 / pretrained 预训练）
    task_modes: dict[Any, str] = {}
    if db is not None:
        try:
            for task in db.list_tasks(500):
                tn = task.get("task_name")
                if not tn:
                    continue
                # 优先使用训练时记录的精确尺寸（actual_model_version，如 YOLOv5lu）；
                # 老任务无该字段时回退解析权重文件名（yolo11n.pt → YOLO11n）
                cfg = task.get("config_json") or "{}"
                try:
                    cfg_data = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
                except Exception:
                    cfg_data = {}
                wp = (cfg_data.get("weights_path") or "").strip()
                ver = cfg_data.get("actual_model_version") or ""
                if not ver:
                    ver = _guess_model_version(Path(wp).name) if wp else ""
                if not ver and task.get("model_version"):
                    ver = task["model_version"]  # 回退到模型大类
                family = _family_of_task(task)
                key: Any = (family, tn) if family else tn
                # 仅在解析出版本时写入，避免空值挡住下方 config.json 文件兜底；
                # 同名任务重跑会产生多条数据库记录（list_tasks 最新在前），用 setdefault 取最新记录的值
                if ver:
                    task_versions.setdefault(key, ver)
                mode = _mode_from_weights(wp)
                if mode:
                    task_modes.setdefault(key, mode)
        except Exception:
            pass
    # 兜底：数据库不可用/记录缺失时，直接从 runs/tasks/<任务>/config.json 读权重名（不覆盖 db 结果）
    try:
        for cfg_path in sorted((base_dir / "runs" / "tasks").glob("*/config.json")):
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tn = cfg.get("task_name") or cfg_path.parent.name
            wp = (cfg.get("weights_path") or "").strip()
            ver = cfg.get("actual_model_version") or ""
            if not ver:
                ver = _guess_model_version(Path(wp).name) if wp else ""
            if not ver and cfg.get("model_version"):
                ver = cfg["model_version"]
            family = str(cfg.get("model_version") or "").strip()
            key = (family, tn) if family else tn
            if ver:
                task_versions.setdefault(key, ver)
            mode = _mode_from_weights(wp)
            if mode:
                task_modes.setdefault(key, mode)
    except Exception:
        pass

    if uploads.exists():
        for path in sorted(uploads.iterdir()):
            if _is_model_dir(path):
                item = _entry_info(base_dir, path, "upload", path.name)
                item["model_version"] = _model_version_for(path, "upload", task_versions)
                items.append(item)
            elif path.is_file() and path.suffix.lower() in FILE_EXTS:
                item = _entry_info(base_dir, path, "upload", path.name)
                item["model_version"] = _model_version_for(path, "upload", task_versions)
                items.append(item)

    for runs in runs_list:
        if not runs.exists():
            continue
        # 定位所有 weights 目录并递归扫描任意层级：rknn-toolkit2 会把产物写成
        # <名称>/best.rknn 嵌套目录，仅扫 weights/* 单层会漏掉这类自定义名称的转换产物
        for weights_dir in sorted(runs.rglob("weights")):
            if not weights_dir.is_dir():
                continue
            task_name = weights_dir.parent.name
            family = weights_dir.parent.parent.name  # runs/<大类>/<任务>/weights
            for path in sorted(weights_dir.rglob("*")):
                if path.is_dir():
                    # 目录型产物（openvino/ncnn 的 _model 目录）整体作为一个条目
                    if not _is_model_dir(path):
                        continue
                    item = _entry_info(base_dir, path, "training", f"{task_name}/{path.name}")
                    item["model_version"] = _model_version_for(path, "training", task_versions, task_name)
                    item["training_mode"] = task_modes.get((family, task_name), "") or task_modes.get(task_name, "")
                    items.append(item)
                    continue
                if not (path.is_file() and path.suffix.lower() in FILE_EXTS):
                    continue
                # _model 目录内部文件已随目录整体展示，跳过避免重复条目
                inside_marker = False
                for parent in path.parents:
                    if parent == weights_dir:
                        break
                    if parent.name.endswith(DIR_MARKER) and parent.is_dir():
                        inside_marker = True
                        break
                if inside_marker:
                    continue
                rel = str(path.relative_to(base_dir)).replace("\\", "/")
                if any(item["path"] == rel for item in items):
                    continue
                # 名称取相对 weights 的路径：直接产物 <任务>/<文件>，嵌套产物 <任务>/<目录>/<文件>
                sub = str(path.relative_to(weights_dir)).replace("\\", "/")
                item = _entry_info(base_dir, path, "training", f"{task_name}/{sub}")
                item["model_version"] = _model_version_for(path, "training", task_versions, task_name)
                item["training_mode"] = task_modes.get((family, task_name), "") or task_modes.get(task_name, "")
                items.append(item)

    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return items
