"""模型转换管理器：在独立子进程中执行 YOLO export，状态持久化到 state 文件。

- 每个转换任务独立子进程，Web 服务重启不影响转换进行；
- 服务端通过 job_id 查询 state 文件获取进度；
- stdout 日志透传写入 state 文件，便于前端展示。
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import ConvertStartRequest


class ConvertManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.converted_dir = base_dir / "state" / "converted"
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}

    def _state_path(self, job_id: str) -> Path:
        return self.converted_dir / f"{job_id}.json"

    def start(self, req: ConvertStartRequest) -> dict[str, Any]:
        job_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + str(int(time.time() * 1000))[-4:]
        config_path = self.converted_dir / f"{job_id}.config.json"
        state_path = self._state_path(job_id)
        payload = req.model_dump()
        payload["base_dir"] = str(self.base_dir)
        config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        state_path.write_text(
            json.dumps({"state": "queued", "job_id": job_id, "message": "转换任务已排队"}, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "server.convert_worker",
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
            state_path.write_text(
                json.dumps({"state": "error", "message": f"无法启动转换进程：{exc}"}, ensure_ascii=False),
                encoding="utf-8",
            )
            raise RuntimeError(f"无法启动转换进程：{exc}")
        with self._lock:
            self._procs[job_id] = proc
        threading.Thread(target=self._watch, args=(proc, state_path, job_id), daemon=True).start()
        return {"job_id": job_id, "state": "queued"}

    def stop(self, job_id: str) -> dict[str, Any]:
        """取消转换任务：终止子进程并标记 stopped。"""
        with self._lock:
            proc = self._procs.pop(job_id, None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()  # SIGTERM
            except Exception:
                pass
            try:
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()  # 超时强杀（SIGKILL）
                except Exception:
                    pass
        # 兜底：无论进程是否已退出，标记状态为 stopped
        state_path = self._state_path(job_id)
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["state"] = "stopped"
        data["message"] = "转换已取消"
        state_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"ok": True}

    def _watch(self, proc: subprocess.Popen, state_path: Path, job_id: str) -> None:
        if proc.stdout is not None:
            for line in proc.stdout:
                text = line.strip()
                if not text:
                    continue
                try:
                    data = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                logs = data.get("logs", [])
                logs.append({"message": text, "level": "info"})
                del logs[:-200]
                data["logs"] = logs
                state_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        proc.wait()
        with self._lock:
            self._procs.pop(job_id, None)

    def status(self, job_id: str) -> dict[str, Any]:
        path = self._state_path(job_id)
        if not path.exists():
            return {"state": "error", "message": "任务不存在"}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"state": "error", "message": "状态读取失败"}
