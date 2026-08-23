from __future__ import annotations

import asyncio
import json
import os
import re
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


def _parse_gpus(device: str) -> list[int]:
    """从 device 参数（'cpu' / '0' / '0,1'）解析出 GPU 编号列表。"""
    device = (device or "").strip().lower()
    if not device or device == "cpu":
        return []
    out: list[int] = []
    for part in device.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def _safe_task_dirname(name: str) -> str:
    """任务名转为安全目录名（仅保留字母数字 . _ -，其余替换为 _）。"""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name or "task") or "task"


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


@dataclass
class JobInfo:
    """单个训练任务的全部运行状态（进程 / 状态文件 / 快照 / 监控线程）。"""

    task_name: str
    model_version: str
    dataset: str
    device: str
    gpus: list[int]
    state_path: Path
    config_path: Path
    proc: Optional[subprocess.Popen] = None
    snapshot: TrainingSnapshot = field(default_factory=TrainingSnapshot)
    supervisor: Optional[threading.Thread] = None
    stop_supervisor: threading.Event = field(default_factory=threading.Event)
    last_state_mtime: float = 0.0
    last_log_id: int = 0
    task_id: Optional[int] = None


class TrainingManager:
    """训练任务管理器（多任务并行）。

    每个任务在独立子进程（server.train_worker）中运行：
    - Web 服务崩溃 / 重启 / 重新部署不会中断训练；
    - 子进程每轮 epoch 将进度原子写入 runs/tasks/<task>/state.json；
    - 本管理器为每个任务启动独立 supervisor 轮询 state 文件，将进度推送给 WebSocket 客户端；
    - 多 GPU 环境下允许多个任务并行，但同一张 GPU 不能同时被两个任务占用（GPU 占用规则）。
    """

    def __init__(self, base_dir: Path, db=None) -> None:
        self.base_dir = base_dir
        self.uploads_dir = base_dir / "uploads"
        self.runs_dir = base_dir / "runs"
        self.tasks_dir = self.runs_dir / "tasks"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._db = db
        self._lock = threading.Lock()
        self._jobs: dict[str, JobInfo] = {}  # task_name -> JobInfo
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

    def _emit_log(self, message: str, level: str = "info", task_name: str = "") -> None:
        self._emit({"type": "log", "message": message, "level": level, "task_name": task_name})

    def _emit_metrics(self, job: JobInfo) -> None:
        snap = job.snapshot
        self._emit(
            {
                "type": "metrics",
                "task_name": job.task_name,
                "state": snap.state.value,
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

    def _update_snapshot(self, job: JobInfo, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(job.snapshot, key, value)
        self._emit_metrics(job)

    # ------------------------------------------------------------------ #
    # 任务列表 / GPU 占用
    # ------------------------------------------------------------------ #
    def _job_state_path(self, task_name: str) -> Path:
        return self.tasks_dir / _safe_task_dirname(task_name) / "state.json"

    def _job_config_path(self, task_name: str) -> Path:
        return self.tasks_dir / _safe_task_dirname(task_name) / "config.json"

    def _read_state(self, job: JobInfo) -> dict[str, Any]:
        if not job.state_path.exists():
            return {}
        try:
            return json.loads(job.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def list_jobs(self) -> list[JobInfo]:
        with self._lock:
            return list(self._jobs.values())

    def gpu_usage(self) -> dict[str, str]:
        """返回 GPU 占用表：{gpu_id: task_name}，仅统计运行中任务。"""
        usage: dict[str, str] = {}
        with self._lock:
            for job in self._jobs.values():
                if job.snapshot.state == TrainingState.RUNNING:
                    for g in job.gpus:
                        usage[str(g)] = job.task_name
        return usage

    def get_status(self) -> dict[str, Any]:
        """返回全部任务状态 + GPU 占用表（供训练看板与 GPU 选择占用展示）。"""
        tasks: list[dict[str, Any]] = []
        running = False
        for job in self.list_jobs():
            snap = job.snapshot
            if snap.state == TrainingState.RUNNING:
                running = True
            tasks.append(
                {
                    "task_name": job.task_name,
                    "model_version": job.model_version,
                    "dataset": job.dataset,
                    "device": job.device,
                    "gpus": job.gpus,
                    "state": snap.state.value,
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
                    "logs": self._read_state(job).get("logs", [])[-80:],
                }
            )
        return {"state": "running" if running else "idle", "tasks": tasks, "gpu_usage": self.gpu_usage()}

    def is_running(self) -> bool:
        with self._lock:
            return any(j.snapshot.state == TrainingState.RUNNING for j in self._jobs.values())

    def get_recent_logs(self, limit: int = 300) -> list[dict[str, Any]]:
        """返回最近 N 条训练日志（兼容旧调用；多任务下合并各任务日志）。"""
        logs: list[dict[str, Any]] = []
        for job in self.list_jobs():
            logs.extend(self._read_state(job).get("logs", [])[-limit:])
        return logs[-limit:]

    # ------------------------------------------------------------------ #
    # 启动 / 停止
    # ------------------------------------------------------------------ #
    def _resolve_setting_dir(self, value: str) -> Optional[Path]:
        """将设置中的目录解析为绝对路径；空值或非法路径返回 None。"""
        value = (value or "").strip()
        if not value:
            return None
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.base_dir / p
        try:
            return p.resolve()
        except (ValueError, OSError):
            return None

    def _resolve_worker_python(self, model_version: str) -> str:
        """优先使用对应模型的 Conda 环境 Python；未找到时回退到当前解释器（venv）。"""
        try:
            from .conda_env import get_conda_python, resolve_env_for_model

            active = ""
            if self._db is not None:
                active = self._db.get_setting("active_conda_env", "")
            env_name = resolve_env_for_model(model_version, active)
            if env_name:
                python_exe = get_conda_python(env_name)
                if python_exe:
                    return python_exe
        except Exception:
            pass
        return sys.executable

    def _check_gpu_conflict(self, gpus: list[int]) -> Optional[str]:
        """校验 GPU 占用冲突，返回冲突提示；无冲突返回 None。"""
        usage = self.gpu_usage()
        for g in gpus:
            owner = usage.get(str(g))
            if owner:
                return f"GPU {g} 已被任务「{owner}」占用，请选择其他 GPU"
        return None

    def start(self, config: TrainingStartRequest) -> None:
        with self._lock:
            if config.task_name in self._jobs:
                raise RuntimeError(f"任务名称「{config.task_name}」已存在，请更换任务名称")
        gpus = _parse_gpus(config.device)
        conflict = self._check_gpu_conflict(gpus)
        if conflict:
            raise RuntimeError(conflict)

        payload = config.model_dump()
        payload["base_dir"] = str(self.base_dir)
        # 训练输出目录与数据集配置目录：优先使用设置中的自定义目录，否则默认项目内 runs / datasets
        if self._db is not None:
            runs_setting = (self._db.get_setting("runs_dir", "") or "").strip()
            cfg_setting = (self._db.get_setting("datasets_cfg_dir", "") or "").strip()
        else:
            runs_setting, cfg_setting = "", ""
        project_dir = self._resolve_setting_dir(runs_setting) or self.runs_dir
        cfg_dir = self._resolve_setting_dir(cfg_setting) or (self.base_dir / "datasets")
        project_dir.mkdir(parents=True, exist_ok=True)
        payload["project_dir"] = str(project_dir)
        payload["datasets_cfg_dir"] = str(cfg_dir)

        state_path = self._job_state_path(config.task_name)
        config_path = self._job_config_path(config.task_name)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if state_path.exists():
            state_path.unlink()
        try:
            proc = subprocess.Popen(
                [
                    self._resolve_worker_python(config.model_version),
                    "-m",
                    "server.train_worker",
                    "--config",
                    str(config_path),
                    "--state",
                    str(state_path),
                ],
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise RuntimeError(f"无法启动训练进程：{exc}")

        job = JobInfo(
            task_name=config.task_name,
            model_version=config.model_version,
            dataset=config.dataset,
            device=config.device,
            gpus=gpus,
            state_path=state_path,
            config_path=config_path,
            proc=proc,
        )
        job.snapshot.task_name = config.task_name
        job.snapshot.state = TrainingState.RUNNING
        job.snapshot.total_epochs = config.epochs
        job.snapshot.message = "训练任务已启动"
        with self._lock:
            self._jobs[config.task_name] = job
        self._emit_log(f"任务已提交：{config.task_name}", "ok", config.task_name)
        self._emit_metrics(job)
        # 任务落库（失败不影响训练本身）
        if self._db is not None:
            try:
                job.task_id = self._db.create_task(
                    config.task_name, config.model_version, config.dataset, config.device, payload
                )
                out = project_dir / config.task_name
                try:
                    out_rel = str(out.relative_to(self.base_dir)).replace("\\", "/")
                except ValueError:
                    out_rel = str(out)
                self._db.update_task(
                    job.task_id,
                    state="running",
                    total_epochs=config.epochs,
                    output_dir=out_rel,
                    message="训练任务已启动",
                )
            except Exception:
                job.task_id = None
        self._start_stdout_reader(job)
        self._start_supervisor(job)

    def stop(self, task_name: str = "") -> None:
        job = None
        with self._lock:
            if task_name:
                job = self._jobs.get(task_name)
            else:
                # 兼容旧调用：未指定任务名时停止任意一个运行中的任务
                for j in self._jobs.values():
                    if j.snapshot.state == TrainingState.RUNNING:
                        job = j
                        break
        if job is None:
            return
        worker = job.proc
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
            except Exception:
                pass
        else:
            data = self._read_state(job)
            pid = data.get("pid")
            if pid and _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        self._emit_log("正在停止训练，保存当前检查点…", "ok", job.task_name)

    # ------------------------------------------------------------------ #
    # 快照同步
    # ------------------------------------------------------------------ #
    def _apply_state(self, job: JobInfo, data: dict[str, Any]) -> None:
        with self._lock:
            for key in (
                "task_name", "epoch", "total_epochs", "box_loss", "val_loss",
                "map50", "progress", "eta_seconds", "message",
                "train_points", "val_points",
            ):
                if key in data:
                    setattr(job.snapshot, key, data[key])
            state_str = data.get("state")
            if state_str:
                job.snapshot.state = STATE_TO_ENUM.get(state_str, TrainingState.ERROR)
        self._emit_metrics(job)
        for entry in data.get("logs") or []:
            if entry.get("id", 0) > job.last_log_id:
                job.last_log_id = entry["id"]
                self._emit_log(entry.get("message", ""), entry.get("level", "info"), job.task_name)
        # 同步训练指标到数据库
        if self._db is not None and job.task_id:
            try:
                snap = job.snapshot
                self._db.update_task(
                    job.task_id,
                    state=snap.state.value,
                    epoch=snap.epoch,
                    total_epochs=snap.total_epochs,
                    box_loss=snap.box_loss,
                    val_loss=snap.val_loss,
                    map50=snap.map50,
                    progress=snap.progress,
                    eta_seconds=snap.eta_seconds,
                    message=snap.message,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 子进程 stdout 转发（独立线程，避免阻塞轮询）
    # ------------------------------------------------------------------ #
    def _start_stdout_reader(self, job: JobInfo) -> None:
        if job.proc is None or job.proc.stdout is None:
            return

        def read() -> None:
            proc = job.proc
            if proc is None:
                return
            for line in proc.stdout:
                text = line.strip()
                if text:
                    self._emit_log(text, "info", job.task_name)

        threading.Thread(target=read, daemon=True).start()

    # ------------------------------------------------------------------ #
    # 轮询 supervisor：state 文件 + 进程存活（每个任务独立）
    # ------------------------------------------------------------------ #
    def _start_supervisor(self, job: JobInfo) -> None:
        if job.supervisor and job.supervisor.is_alive():
            return
        job.stop_supervisor.clear()
        job.supervisor = threading.Thread(target=self._supervisor_loop, args=(job,), daemon=True)
        job.supervisor.start()

    def _supervisor_loop(self, job: JobInfo) -> None:
        while not job.stop_supervisor.is_set():
            time.sleep(2)
            try:
                if job.state_path.exists():
                    mtime = job.state_path.stat().st_mtime
                    if mtime != job.last_state_mtime:
                        job.last_state_mtime = mtime
                        self._apply_state(job, self._read_state(job))
            except Exception:
                pass
            # 本进程启动的子进程退出
            worker = job.proc
            if worker is not None and worker.poll() is not None:
                self._handle_worker_exit(job, worker.returncode)
                self._drop_job(job.task_name)
                break
            # 服务重启后接管场景：无句柄，通过 state + pid 判断
            if worker is None:
                data = self._read_state(job)
                state_str = data.get("state")
                pid = data.get("pid")
                if state_str == "running" and pid and not _pid_alive(pid):
                    self._update_snapshot(job, state=TrainingState.INTERRUPTED, message="训练进程已中断")
                    self._emit_log("训练进程已中断", "error", job.task_name)
                    self._drop_job(job.task_name)
                    break
                if state_str in ("completed", "stopped", "error", "interrupted"):
                    self._apply_state(job, data)
                    self._drop_job(job.task_name)
                    break

    def _drop_job(self, task_name: str) -> None:
        with self._lock:
            self._jobs.pop(task_name, None)

    def _handle_worker_exit(self, job: JobInfo, returncode: int) -> None:
        data = self._read_state(job)
        final_state = data.get("state")
        if final_state in ("completed", "stopped", "error"):
            self._apply_state(job, data)
            label = {"completed": "训练完成", "stopped": "训练已停止", "error": "训练失败"}[final_state]
            self._emit_log(f"训练进程已退出（{label}）", "ok" if final_state != "error" else "error", job.task_name)
        else:
            self._update_snapshot(job, state=TrainingState.INTERRUPTED, message="训练进程异常退出")
            self._emit_log("训练进程异常退出（可能被强制终止）", "error", job.task_name)
        if self._db is not None and job.task_id:
            try:
                from .database import _utc_now
                self._db.update_task(
                    job.task_id,
                    state=job.snapshot.state.value,
                    finished_at=_utc_now(),
                    message=job.snapshot.message,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 服务重启后恢复监控
    # ------------------------------------------------------------------ #
    def _attach_recovered_task(self, job: JobInfo) -> None:
        """将恢复的任务关联到数据库记录。"""
        if self._db is None or job.task_id:
            return
        try:
            for task in self._db.list_tasks(50):
                if task["task_name"] == job.task_name and task["state"] in ("queued", "running"):
                    job.task_id = task["id"]
                    return
        except Exception:
            pass

    def recover(self) -> None:
        """扫描 runs/tasks/* 下所有 state 文件，接管仍在运行的训练任务。"""
        if not self.tasks_dir.exists():
            return
        for state_path in sorted(self.tasks_dir.glob("*/state.json")):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            task_name = data.get("task_name") or state_path.parent.name
            config_path = state_path.parent / "config.json"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
            state_str = data.get("state")
            pid = data.get("pid")
            if state_str == "running" and pid and _pid_alive(pid):
                job = JobInfo(
                    task_name=task_name,
                    model_version=config.get("model_version", ""),
                    dataset=config.get("dataset", ""),
                    device=config.get("device", "cpu"),
                    gpus=_parse_gpus(config.get("device", "cpu")),
                    state_path=state_path,
                    config_path=config_path,
                    proc=None,
                )
                job.last_state_mtime = state_path.stat().st_mtime
                with self._lock:
                    self._jobs[task_name] = job
                self._attach_recovered_task(job)
                self._apply_state(job, data)
                self._emit_log(f"检测到正在运行的训练任务「{task_name}」，已恢复监控", "ok", task_name)
                self._start_supervisor(job)
