from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .schemas import TrainingStartRequest, TrainingState

STATE_TO_ENUM = {
    "idle": TrainingState.IDLE,
    "queued": TrainingState.QUEUED,
    "running": TrainingState.RUNNING,
    "completed": TrainingState.COMPLETED,
    "stopped": TrainingState.STOPPED,
    "error": TrainingState.ERROR,
    "interrupted": TrainingState.INTERRUPTED,
}


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass
class TrainingSnapshot:
    state: TrainingState = TrainingState.IDLE
    task_name: str = ""
    epoch: int = 0
    total_epochs: int = 0
    box_loss: Optional[float] = None
    val_loss: Optional[float] = None
    map50: Optional[float] = None
    progress: float = 0.0
    eta_seconds: Optional[int] = None
    message: str = ""
    train_points: list[tuple[float, float]] = field(default_factory=list)
    val_points: list[tuple[float, float]] = field(default_factory=list)


class TrainingManager:
    """训练任务管理器。

    训练在独立子进程（server.train_worker）中运行：
    - Web 服务崩溃 / 重启 / 重新部署不会中断训练；
    - 子进程每轮 epoch 将进度原子写入 runs/worker_state.json；
    - 本管理器轮询 state 文件，将进度推送给 WebSocket 客户端；
    - 服务重启后调用 recover() 自动接管仍在运行的训练任务。
    """

    def __init__(self, base_dir: Path, db=None) -> None:
        self.base_dir = base_dir
        self.uploads_dir = base_dir / "uploads"
        self.runs_dir = base_dir / "runs"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.worker_state_path = self.runs_dir / "worker_state.json"
        self.worker_config_path = self.runs_dir / "worker_config.json"
        self._db = db
        self._task_id: Optional[str] = None
        self._lock = threading.Lock()
        self._worker: Optional[subprocess.Popen] = None
        self._supervisor: Optional[threading.Thread] = None
        self._stop_supervisor = threading.Event()
        self._last_state_mtime = 0.0
        self._last_log_id = 0
        self.snapshot = TrainingSnapshot()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def _emit(self, payload: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            try:
                listener(payload)
            except Exception:
                pass

    def _emit_log(self, message: str, level: str = "info") -> None:
        self._emit({"type": "log", "message": message, "level": level})

    def _emit_metrics(self) -> None:
        snap = self.snapshot
        self._emit(
            {
                "type": "metrics",
                "state": snap.state.value,
                "task_name": snap.task_name,
                "epoch": snap.epoch,
                "total_epochs": snap.total_epochs,
                "box_loss": snap.box_loss,
                "val_loss": snap.val_loss,
                "map50": snap.map50,
                "progress": snap.progress,
                "eta_seconds": snap.eta_seconds,
                "train_points": snap.train_points,
                "val_points": snap.val_points,
                "message": snap.message,
            }
        )

    def _update_snapshot(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self.snapshot, key, value)
        self._emit_metrics()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            snap = self.snapshot
            return {
                "state": snap.state.value,
                "task_name": snap.task_name,
                "epoch": snap.epoch,
                "total_epochs": snap.total_epochs,
                "box_loss": snap.box_loss,
                "val_loss": snap.val_loss,
                "map50": snap.map50,
                "progress": snap.progress,
                "eta_seconds": snap.eta_seconds,
                "message": snap.message,
                "train_points": snap.train_points,
                "val_points": snap.val_points,
            }

    def is_running(self) -> bool:
        with self._lock:
            return self.snapshot.state == TrainingState.RUNNING

    # ------------------------------------------------------------------ #
    # 启动 / 停止
    # ------------------------------------------------------------------ #
    def start(self, config: TrainingStartRequest) -> None:
        if self.is_running():
            raise RuntimeError("已有训练任务正在运行")
        payload = config.model_dump()
        payload["base_dir"] = str(self.base_dir)
        self.worker_config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if self.worker_state_path.exists():
            self.worker_state_path.unlink()
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "server.train_worker",
                    "--config",
                    str(self.worker_config_path),
                    "--state",
                    str(self.worker_state_path),
                ],
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise RuntimeError(f"无法启动训练进程：{exc}")
        self._worker = proc
        self._last_state_mtime = 0.0
        self._last_log_id = 0
        self._update_snapshot(
            state=TrainingState.RUNNING,
            task_name=config.task_name,
            total_epochs=config.epochs,
            progress=0.0,
            message="训练任务已启动",
            train_points=[],
            val_points=[],
        )
        self._emit_log(f"任务已提交：{config.task_name}", "ok")
        # 任务落库（失败不影响训练本身）
        if self._db is not None:
            try:
                self._task_id = self._db.create_task(
                    config.task_name, config.model_version, config.dataset, config.device, payload
                )
                self._db.update_task(
                    self._task_id,
                    state="running",
                    total_epochs=config.epochs,
                    output_dir=f"runs/{config.task_name}",
                    message="训练任务已启动",
                )
            except Exception:
                self._task_id = None
        self._start_stdout_reader(proc)
        self._start_supervisor()

    def stop(self) -> None:
        if not self.is_running():
            return
        worker = self._worker
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
            except Exception:
                pass
        else:
            data = self._read_state()
            pid = data.get("pid")
            if pid and _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        self._emit_log("正在停止训练，保存当前检查点…", "ok")

    # ------------------------------------------------------------------ #
    # state 文件读取与快照同步
    # ------------------------------------------------------------------ #
    def _read_state(self) -> dict[str, Any]:
        if not self.worker_state_path.exists():
            return {}
        try:
            return json.loads(self.worker_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_recent_logs(self, limit: int = 300) -> list[dict[str, Any]]:
        """返回最近 N 条训练日志（供新连接的客户端补发历史日志）。"""
        return self._read_state().get("logs", [])[-limit:]

    def _apply_state(self, data: dict[str, Any]) -> None:
        with self._lock:
            for key in (
                "task_name", "epoch", "total_epochs", "box_loss", "val_loss",
                "map50", "progress", "eta_seconds", "message",
                "train_points", "val_points",
            ):
                if key in data:
                    setattr(self.snapshot, key, data[key])
            state_str = data.get("state")
            if state_str:
                self.snapshot.state = STATE_TO_ENUM.get(state_str, TrainingState.ERROR)
        self._emit_metrics()
        for entry in data.get("logs") or []:
            if entry.get("id", 0) > self._last_log_id:
                self._last_log_id = entry["id"]
                self._emit_log(entry.get("message", ""), entry.get("level", "info"))
        # 同步训练指标到数据库
        if self._db is not None and self._task_id:
            try:
                self._db.update_task(
                    self._task_id,
                    state=self.snapshot.state.value,
                    epoch=self.snapshot.epoch,
                    total_epochs=self.snapshot.total_epochs,
                    box_loss=self.snapshot.box_loss,
                    val_loss=self.snapshot.val_loss,
                    map50=self.snapshot.map50,
                    progress=self.snapshot.progress,
                    eta_seconds=self.snapshot.eta_seconds,
                    message=self.snapshot.message,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 子进程 stdout 转发（独立线程，避免阻塞轮询）
    # ------------------------------------------------------------------ #
    def _start_stdout_reader(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return

        def read() -> None:
            for line in proc.stdout:
                text = line.strip()
                if text:
                    self._emit_log(text, "info")

        threading.Thread(target=read, daemon=True).start()

    # ------------------------------------------------------------------ #
    # 轮询 supervisor：state 文件 + 进程存活
    # ------------------------------------------------------------------ #
    def _start_supervisor(self) -> None:
        if self._supervisor and self._supervisor.is_alive():
            return
        self._stop_supervisor.clear()
        self._supervisor = threading.Thread(target=self._supervisor_loop, daemon=True)
        self._supervisor.start()

    def _supervisor_loop(self) -> None:
        worker = self._worker
        while not self._stop_supervisor.is_set():
            time.sleep(2)
            try:
                if self.worker_state_path.exists():
                    mtime = self.worker_state_path.stat().st_mtime
                    if mtime != self._last_state_mtime:
                        self._last_state_mtime = mtime
                        self._apply_state(self._read_state())
            except Exception:
                pass
            # 本进程启动的子进程退出
            if worker is not None and worker.poll() is not None:
                self._handle_worker_exit(worker.returncode)
                self._worker = None
                break
            # 服务重启后接管场景：无句柄，通过 state + pid 判断
            if worker is None:
                data = self._read_state()
                state_str = data.get("state")
                pid = data.get("pid")
                if state_str == "running" and pid and not _pid_alive(pid):
                    self._update_snapshot(state=TrainingState.INTERRUPTED, message="训练进程已中断")
                    self._emit_log("训练进程已中断", "error")
                    break
                if state_str in ("completed", "stopped", "error", "interrupted"):
                    self._apply_state(data)
                    break

    def _handle_worker_exit(self, returncode: int) -> None:
        data = self._read_state()
        final_state = data.get("state")
        if final_state in ("completed", "stopped", "error"):
            self._apply_state(data)
            label = {"completed": "训练完成", "stopped": "训练已停止", "error": "训练失败"}[final_state]
            self._emit_log(f"训练进程已退出（{label}）", "ok" if final_state != "error" else "error")
        else:
            self._update_snapshot(state=TrainingState.INTERRUPTED, message="训练进程异常退出")
            self._emit_log("训练进程异常退出（可能被强制终止）", "error")
        if self._db is not None and self._task_id:
            try:
                from .database import _utc_now
                self._db.update_task(
                    self._task_id,
                    state=self.snapshot.state.value,
                    finished_at=_utc_now(),
                    message=self.snapshot.message,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 服务重启后恢复监控
    # ------------------------------------------------------------------ #
    def _attach_recovered_task(self) -> None:
        """服务重启后，将 state 文件对应的训练任务关联到数据库记录。"""
        if self._db is None or self._task_id:
            return
        try:
            if not self.worker_config_path.exists():
                return
            config = json.loads(self.worker_config_path.read_text(encoding="utf-8"))
            for task in self._db.list_tasks(10):
                if task["task_name"] == config.get("task_name") and task["state"] in ("queued", "running"):
                    self._task_id = task["id"]
                    return
        except Exception:
            pass

    def recover(self) -> None:
        data = self._read_state()
        if not data:
            return
        self._attach_recovered_task()
        state_str = data.get("state")
        pid = data.get("pid")
        if state_str == "running" and pid and _pid_alive(pid):
            self._last_log_id = 0
            self._apply_state(data)
            self._emit_log("检测到正在运行的训练任务，已恢复监控", "ok")
            self._start_supervisor()
        elif state_str == "running":
            self._last_log_id = 0
            self._apply_state(data)
            self._update_snapshot(state=TrainingState.INTERRUPTED, message="训练进程已中断（服务重启）")
            self._emit_log("检测到上次训练被中断，进度与检查点已保留", "error")
        elif state_str:
            # 已完成 / 已停止 / 失败 等终态：恢复快照供查看
            self._last_log_id = 0
            self._apply_state(data)
