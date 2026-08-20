from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml

from .database import Database

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
    def __init__(self, base_dir: Path, db: Database) -> None:
        self.base_dir = base_dir
        self.datasets_dir = base_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

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
            return {
                "root_path": data.get("path", ""),
                "train": data.get("train", ""),
                "val": data.get("val", ""),
                "test": data.get("test", ""),
                "class_count": data.get("nc", 0),
                "class_names": names,
                "download": data.get("download", ""),
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
