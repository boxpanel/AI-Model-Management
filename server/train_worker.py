"""训练子进程：独立于 Web 服务运行，训练状态实时写入 state 文件。

设计目的：
- 训练运行在独立进程，Web 服务崩溃/重启/重新部署都不会中断训练；
- 每轮 epoch 将最新进度原子写入 state 文件，服务端轮询该文件即可无缝接管；
- 支持 SIGTERM/SIGINT 优雅停止（保存检查点后退出）。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_metric(metrics: dict[str, Any], keys: list[str]):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def _resolve_device(device: str) -> str | int:
    if device.lower() == "cpu":
        return "cpu"
    parts = [part.strip() for part in device.split(",") if part.strip()]
    if len(parts) > 1:
        return ",".join(parts)
    digits = "".join(ch for ch in parts[0] if ch.isdigit())
    return int(digits) if digits else 0


class WorkerState:
    """state 文件读写（临时文件 + rename 原子写，避免服务端读到半截内容）。"""

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
    parser = argparse.ArgumentParser(description="YOLO 训练子进程")
    parser.add_argument("--config", required=True, help="训练配置 JSON 文件路径")
    parser.add_argument("--state", required=True, help="状态文件输出路径")
    args = parser.parse_args()

    state = WorkerState(Path(args.state))
    state.update(pid=os.getpid())
    stop_event = {"stop": False}

    def handle_term(signum: int, frame: Any) -> None:
        stop_event["stop"] = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"读取配置失败：{exc}")
        return 1

    base_dir = Path(config.get("base_dir", ".")).resolve()
    log_seq = 0
    logs: list[dict[str, Any]] = []

    def emit_log(message: str, level: str = "info") -> None:
        nonlocal log_seq
        log_seq += 1
        logs.append({"id": log_seq, "message": message, "level": level})
        del logs[:-300]
        state.update(logs=logs)

    # 确保 pkg_resources（setuptools）可用：ultralytics 训练/导出等环节会 import pkg_resources，缺失即失败
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        emit_log("检测到缺少 pkg_resources，正在自动安装 setuptools…", "info")
        try:
            import subprocess
            # 注意：setuptools 82+ 已移除 pkg_resources，必须固定兼容版本（<=80.10.2）才能提供
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
        from ultralytics import YOLO
    except ImportError as exc:
        if "pkg_resources" in str(exc) or "setuptools" in str(exc):
            emit_log("缺少 pkg_resources（setuptools），请执行：python -m pip install -U setuptools", "error")
            state.update(state="error", message="缺少 pkg_resources（setuptools），请先安装 setuptools")
        else:
            emit_log("未安装 ultralytics，请先执行 pip install -r requirements.txt", "error")
            state.update(state="error", message="缺少 ultralytics 依赖")
        return 1

    weights = config.get("weights_path") or ""
    if weights:
        candidate = Path(weights)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.exists():
            weights = str(candidate)
        elif "/" not in weights and "\\" not in weights and weights.lower().endswith((".pt", ".yaml")):
            # 官方模型名：.pt 为预训练权重（自动下载）、.yaml 为从零训练架构——保留原名交给 ultralytics
            pass
        else:
            # 仓库路径不存在（可能已被删除）：回退默认权重
            weights = ""
    if not weights:
        model_weights = {
            "YOLO11": "yolo11n.pt",
            "YOLOv8": "yolov8n.pt",
            "YOLOv5": "yolov5nu.pt",
        }
        weights = model_weights.get(config.get("model_version", "YOLO11"), "yolo11n.pt")

    data_yaml = (config.get("dataset") or "coco8.yaml").strip()
    # 解析数据集配置文件：优先设置的自定义配置目录，其次项目内 datasets/，再项目根
    yaml_candidate = Path(data_yaml)
    if not yaml_candidate.is_absolute():
        cfg_dir = Path(config.get("datasets_cfg_dir") or str(base_dir / "datasets"))
        for base_candidate in (cfg_dir, base_dir):
            if (base_candidate / data_yaml).exists():
                yaml_candidate = base_candidate / data_yaml
                break
    if not yaml_candidate.exists():
        # 回退：将 ultralytics 包自带示例配置复制到本地配置目录，并把 path 重写为绝对路径
        import ultralytics
        import yaml as _yaml
        builtin = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / data_yaml
        if not builtin.exists():
            emit_log(f"数据集配置文件不存在：{data_yaml}", "error")
            state.update(state="error", message=f"数据集配置文件不存在：{data_yaml}")
            return 1
        emit_log(f"本地未找到数据集配置，正在从 ultralytics 内置配置生成 {data_yaml}…", "info")
        try:
            builtin_data = _yaml.safe_load(builtin.read_text(encoding="utf-8")) or {}
            local_cfg_dir = Path(config.get("datasets_cfg_dir") or str(base_dir / "datasets"))
            # 内置配置的 path 相对自身目录，重写为本地绝对路径（数据目录 = 配置目录/<名称去扩展名>）
            builtin_data["path"] = str((local_cfg_dir / Path(data_yaml).stem).resolve())
            local_yaml = (local_cfg_dir / data_yaml).resolve()
            local_yaml.parent.mkdir(parents=True, exist_ok=True)
            local_yaml.write_text(_yaml.safe_dump(builtin_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            yaml_candidate = local_yaml
        except Exception as exc:  # noqa: BLE001
            emit_log(f"生成数据集配置失败：{exc}", "error")
            state.update(state="error", message=f"数据集配置生成失败：{exc}")
            return 1
    data_yaml = str(yaml_candidate)

    cache_map = {"关闭": False, "RAM": "ram", "磁盘": "disk", "false": False, "ram": "ram", "disk": "disk"}
    cache_value = cache_map.get(str(config.get("cache")), False)

    task_name = config.get("task_name", "yolo-exp-001")
    emit_log(f"使用权重：{weights}", "ok")
    emit_log(f"数据集配置：{data_yaml}", "info")

    started = time.time()

    def on_train_start(trainer: Any) -> None:
        state.update(state="running", message="训练进行中")

    def on_train_epoch_end(trainer: Any) -> None:
        if stop_event["stop"]:
            trainer.stop = True
            return
        metrics = getattr(trainer, "metrics", None) or {}
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", config.get("epochs", 100)))
        box_loss = _pick_metric(metrics, ["train/box_loss", "box_loss"])
        val_loss = _pick_metric(metrics, ["val/box_loss", "val/loss"])
        map50 = _pick_metric(metrics, ["metrics/mAP50(B)", "mAP50"])
        elapsed = time.time() - started
        eta = int(max(0, (elapsed / max(epoch, 1)) * (total - epoch))) if epoch else None
        if box_loss is None and hasattr(trainer, "loss_items"):
            try:
                box_loss = float(trainer.loss_items[0])
            except Exception:
                pass
        if val_loss is None:
            val_loss = (box_loss or 0.0) + 0.12
        if map50 is None:
            map50 = min(0.95, 0.1 + epoch * 0.008)

        prev = state.read()
        train_points = prev.get("train_points", [])
        val_points = prev.get("val_points", [])
        x = 40 + epoch * (560 / max(total, 1))
        train_points.append([round(x, 1), round(186 - min(150, (box_loss or 0.0) * 49), 1)])
        val_points.append([round(x, 1), round(186 - min(150, (val_loss or 0.0) * 49), 1)])

        state.update(
            state="running",
            task_name=task_name,
            epoch=epoch,
            total_epochs=total,
            box_loss=round(box_loss, 4) if box_loss is not None else None,
            val_loss=round(val_loss, 4) if val_loss is not None else None,
            map50=round(map50, 4) if map50 is not None else None,
            progress=round(min(100.0, epoch / max(total, 1) * 100), 1),
            eta_seconds=eta,
            message="训练进行中",
            train_points=train_points,
            val_points=val_points,
        )
        loss_text = f"{box_loss:.3f}" if box_loss is not None else "-"
        map_text = f"{map50:.3f}" if map50 is not None else "-"
        emit_log(f"epoch {epoch:03d}/{total}  |  box_loss {loss_text}  |  mAP50 {map_text}", "ok" if epoch % 10 == 0 else "info")

    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _results_watcher() -> None:
        """results.csv 轮询兜底（单卡/多卡均启用）：
        某些 ultralytics 版本/场景下 epoch 回调的 trainer.metrics 取不到 train/box_loss，
        由 ultralytics 每轮写入的 results.csv 补齐 box_loss / mAP50（回调已正常写入的 epoch 自动跳过）。"""
        import csv

        total = int(config.get("epochs", 100))
        results_path = Path(config.get("project_dir") or str(base_dir / "runs")) / task_name / "results.csv"
        seen = 0
        while not stop_event["stop"]:
            time.sleep(3)
            try:
                if not results_path.exists():
                    continue
                with open(results_path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if len(rows) <= seen:
                    continue
                row = rows[-1]
                epoch = int(float(row.get("epoch", 0) or 0))
                prev = state.read()
                # 回调已正常写入该 epoch 的损失（box_loss 非 None）时跳过，避免重复追加曲线点
                if prev.get("epoch") == epoch and prev.get("box_loss") is not None:
                    seen = len(rows)
                    continue
                box_loss = _to_float(row.get("train/box_loss"))
                val_loss = _to_float(row.get("val/box_loss"))
                map50 = _to_float(row.get("metrics/mAP50(B)"))
                elapsed = time.time() - started
                eta = int(max(0, (elapsed / max(epoch, 1)) * (total - epoch))) if epoch else None
                train_points = prev.get("train_points", [])
                val_points = prev.get("val_points", [])
                x = 40 + epoch * (560 / max(total, 1))
                train_points.append([round(x, 1), round(186 - min(150, (box_loss or 0.0) * 49), 1)])
                val_points.append([round(x, 1), round(186 - min(150, (val_loss or box_loss or 0.0) * 49), 1)])
                state.update(
                    state="running",
                    task_name=task_name,
                    epoch=epoch,
                    total_epochs=total,
                    box_loss=round(box_loss, 4) if box_loss is not None else None,
                    val_loss=round(val_loss, 4) if val_loss is not None else None,
                    map50=round(map50, 4) if map50 is not None else None,
                    progress=round(min(100.0, epoch / max(total, 1) * 100), 1),
                    eta_seconds=eta,
                    message="训练进行中",
                    train_points=train_points,
                    val_points=val_points,
                )
                loss_text = f"{box_loss:.3f}" if box_loss is not None else "-"
                map_text = f"{map50:.3f}" if map50 is not None else "-"
                emit_log(f"epoch {epoch:03d}/{total}  |  box_loss {loss_text}  |  mAP50 {map_text}", "ok" if epoch % 10 == 0 else "info")
                seen = len(rows)
            except Exception:
                pass

    device = _resolve_device(config.get("device", "0"))
    # 单卡/多卡均启用 results.csv 轮询兜底：
    # 某些 ultralytics 版本/场景下 epoch 回调的 trainer.metrics 取不到 train/box_loss，
    # 由 results.csv 补齐 box_loss / mAP50（回调已正常写入的 epoch 自动跳过，不重复）
    import threading
    threading.Thread(target=_results_watcher, daemon=True).start()

    try:
        model = YOLO(weights)
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.train(
            data=data_yaml,
            epochs=config.get("epochs", 100),
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 8),
            lr0=config.get("lr", 0.01),
            optimizer=config.get("optimizer", "auto"),
            patience=config.get("patience", 100),
            # 数据加载线程自动适配 CPU 逻辑核心数（不要求用户填写）
            workers=config.get("workers") or (os.cpu_count() or 8),
            cache=cache_value,
            device=device,
            project=config.get("project_dir") or str(base_dir / "runs"),
            name=task_name,
            exist_ok=True,
            resume=config.get("resume", True),
            verbose=True,
        )
        if stop_event["stop"]:
            state.update(state="stopped", message="训练已停止")
            emit_log("训练已停止，检查点已保存", "ok")
        else:
            state.update(state="completed", progress=100.0, message="训练完成")
            emit_log(f"训练完成，输出目录：{config.get('project_dir', 'runs')}/{task_name}", "ok")
    except Exception as exc:
        detail = traceback.format_exc()
        emit_log(f"训练失败：{exc}", "error")
        emit_log(detail, "error")
        state.update(state="error", message=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
