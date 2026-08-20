from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

# 模型仓库可识别的单文件扩展名（含转换产物）
FILE_EXTS = {".pt", ".onnx", ".engine", ".tflite", ".torchscript", ".xml", ".rknn"}
# 目录型转换产物（ultralytics 导出为目录：openvino / ncnn / tfjs / saved_model / paddle）
DIR_MARKER = "_model"


def _entry_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _entry_info(base_dir: Path, path: Path, source: str, name: str) -> dict[str, Any]:
    rel = str(path.relative_to(base_dir)).replace("\\", "/")
    return {
        "name": name,
        "path": rel,
        "source": source,
        "size_mb": round(_entry_size(path) / (1024 * 1024), 2),
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and DIR_MARKER in path.name


def list_models(base_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    uploads = base_dir / "uploads"
    runs = base_dir / "runs"

    if uploads.exists():
        for path in sorted(uploads.iterdir()):
            if _is_model_dir(path):
                items.append(_entry_info(base_dir, path, "upload", path.name))
            elif path.is_file() and path.suffix.lower() in FILE_EXTS:
                items.append(_entry_info(base_dir, path, "upload", path.name))

    if runs.exists():
        for path in sorted(runs.glob("**/weights/*")):
            if _is_model_dir(path):
                task_name = path.parent.parent.name
                items.append(_entry_info(base_dir, path, "training", f"{task_name}/{path.name}"))
            elif path.is_file() and path.suffix.lower() in FILE_EXTS:
                rel = str(path.relative_to(base_dir)).replace("\\", "/")
                if any(item["path"] == rel for item in items):
                    continue
                task_name = path.parent.parent.name
                items.append(_entry_info(base_dir, path, "training", f"{task_name}/{path.name}"))

    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return items
