from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

# 模型仓库可识别的单文件扩展名（含转换产物）
FILE_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".xml", ".rknn"}
# 目录型转换产物（ultralytics 导出为目录：openvino / ncnn / tfjs / saved_model / paddle）
DIR_MARKER = "_model"


def _guess_model_version(name: str) -> str:
    """从权重文件名解析模型版本（如 yolo11n.pt → YOLO11n、yolov5nu.pt → YOLOv5nu），无法识别返回空。"""
    m = re.match(r"^yolo(11|v8|v5)([nsmlx])(u)?\.pt$", name.lower())
    if not m:
        return ""
    fam, size, u = m.groups()
    prefix = {"11": "YOLO11", "v8": "YOLOv8", "v5": "YOLOv5"}[fam]
    return prefix + size + (u or "")


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

    # 任务名 → 模型版本 映射（训练产出 best.pt 从数据库查询实际使用的模型版本）
    task_versions: dict[str, str] = {}
    if db is not None:
        try:
            for task in db.list_tasks(500):
                if task.get("task_name") and task.get("model_version"):
                    task_versions[task["task_name"]] = task["model_version"]
        except Exception:
            pass

    if uploads.exists():
        for path in sorted(uploads.iterdir()):
            if _is_model_dir(path):
                items.append(_entry_info(base_dir, path, "upload", path.name))
            elif path.is_file() and path.suffix.lower() in FILE_EXTS:
                item = _entry_info(base_dir, path, "upload", path.name)
                item["model_version"] = _guess_model_version(path.name)
                items.append(item)

    for runs in runs_list:
        if not runs.exists():
            continue
        for path in sorted(runs.glob("**/weights/*")):
            if _is_model_dir(path):
                task_name = path.parent.parent.name
                item = _entry_info(base_dir, path, "training", f"{task_name}/{path.name}")
                item["model_version"] = task_versions.get(task_name, "")
                items.append(item)
            elif path.is_file() and path.suffix.lower() in FILE_EXTS:
                rel = str(path.relative_to(base_dir)).replace("\\", "/")
                if any(item["path"] == rel for item in items):
                    continue
                task_name = path.parent.parent.name
                item = _entry_info(base_dir, path, "training", f"{task_name}/{path.name}")
                item["model_version"] = task_versions.get(task_name, "")
                items.append(item)

    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return items
