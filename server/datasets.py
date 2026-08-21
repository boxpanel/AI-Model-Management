from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml

from .database import Database

# 数据集 YAML 的标准键（ultralytics 官方），其余键作为高级参数保留
_STANDARD_KEYS = {"path", "train", "val", "test", "names", "nc", "download", "kpt_shape", "flip_idx"}

# COCO 80 类（ultralytics 官方顺序），用于自动生成 coco8/coco128 内置预设 yaml
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# 前端固定预设 traffic-object-v3 的类别
TRAFFIC_NAMES = ["person", "car", "truck", "bus"]

BUILTIN_DATASETS = [
    {
        "name": "coco8.yaml",
        "yaml_path": "coco8.yaml",
        "root_path": "datasets/coco8",
        "class_count": 80,
        "split": "4 / 4 张",
        "builtin": True,
        "train": "images/train",
        "val": "images/val",
        "test": "",
        "names": [],
        "download": "",
    },
    {
        "name": "coco128.yaml",
        "yaml_path": "coco128.yaml",
        "root_path": "datasets/coco128",
        "class_count": 80,
        "split": "128 / 128 张",
        "builtin": True,
        "train": "images/train2017",
        "val": "images/train2017",
        "test": "",
        "names": [],
        "download": "",
    },
]


class DatasetService:
    def __init__(self, base_dir: Path, db: Database, datasets_dir: Path | None = None) -> None:
        self.base_dir = base_dir
        self.datasets_dir = (datasets_dir or (base_dir / "datasets")).resolve()
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self._ensure_builtin_yamls()

    def _ensure_builtin_yamls(self) -> None:
        """启动时自动生成内置预设 yaml（缺失时），保证前端预设选中后即可直接训练。"""
        presets = [
            ("coco8.yaml", "coco8", "images/train", "images/val", "", 80, COCO_NAMES, "https://ultralytics.com/assets/coco8.zip"),
            ("coco128.yaml", "coco128", "images/train2017", "images/train2017", "", 80, COCO_NAMES, "https://ultralytics.com/assets/coco128.zip"),
            ("traffic-object-v3.yaml", "traffic-object-v3", "images/train", "images/val", "images/test", 4, TRAFFIC_NAMES, ""),
        ]
        for name, data_dir, train, val, test, nc, names, download in presets:
            target = self.datasets_dir / name
            if target.exists():
                continue
            payload = {
                # path 用绝对路径，避免 ultralytics 按 yaml 所在目录解析相对路径时定位错误
                "path": str(self.datasets_dir / data_dir),
                "train": train,
                "val": val,
                "test": test,
                "nc": nc,
                "names": names,
            }
            if download:
                payload["download"] = download
            try:
                target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
            except OSError:
                pass

    def list_all(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        index: dict[str, int] = {}
        for item in BUILTIN_DATASETS:
            entry = dict(item)
            file_path = self.datasets_dir / item["name"]
            if file_path.exists():
                info = self._read_yaml_full(file_path)
                entry["root_path"] = info.get("root_path") or entry["root_path"]
                entry["class_count"] = info.get("class_count") or entry["class_count"]
                entry["train"] = info.get("train")
                entry["val"] = info.get("val")
                entry["test"] = info.get("test")
                entry["names"] = info.get("class_names")
                entry["download"] = info.get("download")
            items.append(entry)
            index[item["name"]] = len(items) - 1
        for row in self.db.list_datasets():
            if row["name"] in index:
                continue
            items.append(
                {
                    "name": row["name"],
                    "yaml_path": row["yaml_path"],
                    "root_path": row["root_path"],
                    "class_count": row["class_count"],
                    "split": "自定义",
                    "builtin": False,
                    "train": "",
                    "val": "",
                    "test": "",
                    "names": [],
                    "download": "",
                    "created_at": row["created_at"],
                }
            )
            index[row["name"]] = len(items) - 1
        # 以 datasets/*.yaml 文件内容为准，增强（或新增）各数据集条目的训练参数
        for path in sorted(self.datasets_dir.glob("*.yaml")):
            info = self._read_yaml_full(path)
            rel = str(path.relative_to(self.base_dir)).replace("\\", "/")
            extra_yaml = ""
            if info.get("extra"):
                extra_yaml = yaml.safe_dump(info["extra"], allow_unicode=True, sort_keys=False).strip()
            if path.name in index:
                entry = items[index[path.name]]
                entry["yaml_path"] = rel
                entry["root_path"] = info.get("root_path") or entry.get("root_path")
                entry["class_count"] = info.get("class_count") or entry.get("class_count")
                entry["train"] = info.get("train")
                entry["val"] = info.get("val")
                entry["test"] = info.get("test")
                entry["names"] = info.get("class_names")
                entry["download"] = info.get("download")
                entry["kpt_shape"] = info.get("kpt_shape", "")
                entry["flip_idx"] = info.get("flip_idx", "")
                entry["extra_yaml"] = extra_yaml
            else:
                items.append(
                    {
                        "name": path.name,
                        "yaml_path": rel,
                        "root_path": info.get("root_path", ""),
                        "class_count": info.get("class_count", 0),
                        "split": "文件",
                        "builtin": False,
                        "train": info.get("train", ""),
                        "val": info.get("val", ""),
                        "test": info.get("test", ""),
                        "names": info.get("class_names", []),
                        "download": info.get("download", ""),
                        "kpt_shape": info.get("kpt_shape", ""),
                        "flip_idx": info.get("flip_idx", ""),
                        "extra_yaml": extra_yaml,
                    }
                )
                index[path.name] = len(items) - 1
        return items

    def _read_yaml_info(self, path: Path) -> dict[str, Any]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {"path": data.get("path", ""), "nc": data.get("nc", 0)}
        except Exception:
            return {}

    def _read_yaml_full(self, path: Path) -> dict[str, Any]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = data.get("names", [])
            if isinstance(names, dict):
                names = list(names.values())
            if not isinstance(names, list):
                names = []
            extra = {k: v for k, v in data.items() if k not in _STANDARD_KEYS}
            return {
                "root_path": data.get("path", ""),
                "train": data.get("train", ""),
                "val": data.get("val", ""),
                "test": data.get("test", ""),
                "class_count": data.get("nc", 0),
                "class_names": names,
                "download": data.get("download", ""),
                "kpt_shape": ",".join(str(x) for x in data.get("kpt_shape") or []),
                "flip_idx": ",".join(str(x) for x in data.get("flip_idx") or []),
                "extra": extra,
            }
        except Exception:
            return {}

    def create_yaml(
        self,
        name: str,
        root_path: str,
        train_path: str,
        val_path: str,
        class_count: int,
        class_names: list[str] | None = None,
        test_path: str = "",
        download_url: str = "",
        kpt_shape: str = "",
        flip_idx: str = "",
    ) -> dict[str, str]:
        if not name.endswith(".yaml"):
            name = f"{name}.yaml"
        # 名称校验：拒绝路径分隔符与隐藏文件，防止写入数据集目录之外
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("非法名称：不能包含路径分隔符")
        target = (self.datasets_dir / name).resolve()
        if not target.is_relative_to(self.datasets_dir.resolve()):
            raise ValueError("非法名称")
        names = class_names or [f"class_{index}" for index in range(class_count)]
        payload: dict[str, Any] = {
            "path": root_path,
            "train": train_path,
            "val": val_path,
            "nc": class_count,
            "names": names,
        }
        if test_path:
            payload["test"] = test_path
        if download_url:
            payload["download"] = download_url
        if kpt_shape.strip():
            payload["kpt_shape"] = [int(x) for x in kpt_shape.replace("，", ",").split(",") if x.strip()]
        if flip_idx.strip():
            payload["flip_idx"] = [int(x) for x in flip_idx.replace("，", ",").split(",") if x.strip()]
        target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        rel = str(target.relative_to(self.base_dir.resolve())).replace("\\", "/")
        self.db.register_dataset(name=name, yaml_path=rel, root_path=root_path, class_count=class_count)
        return {"name": name, "yaml_path": rel}

    def delete_yaml(self, name: str) -> dict[str, Any]:
        removed_file = False
        target = self.datasets_dir / name
        if target.exists():
            target.unlink()
            removed_file = True
        self.db.delete_dataset(name)
        return {"removed_file": removed_file, "builtin": name in {item["name"] for item in BUILTIN_DATASETS}}
