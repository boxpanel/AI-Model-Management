from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    device TEXT NOT NULL,
                    state TEXT NOT NULL,
                    epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER DEFAULT 0,
                    box_loss REAL,
                    val_loss REAL,
                    map50 REAL,
                    progress REAL DEFAULT 0,
                    eta_seconds INTEGER,
                    message TEXT,
                    config_json TEXT NOT NULL,
                    output_dir TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    yaml_path TEXT NOT NULL,
                    root_path TEXT,
                    class_count INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('max_parallel_jobs', '1')"
            )
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES ('active_conda_env', '')"
            )
            conn.commit()

    def get_setting(self, key: str, default: str = "") -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()

    def get_settings(self) -> dict[str, str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            return {row["key"]: row["value"] for row in rows}

    def create_task(self, task_name: str, model_version: str, dataset: str, device: str, config: dict[str, Any]) -> str:
        task_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(
                    id, task_name, model_version, dataset, device, state, total_epochs,
                    config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    task_id,
                    task_name,
                    model_version,
                    dataset,
                    device,
                    int(config.get("epochs", 100)),
                    json.dumps(config, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()
        return task_id

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "state",
            "epoch",
            "total_epochs",
            "box_loss",
            "val_loss",
            "map50",
            "progress",
            "eta_seconds",
            "message",
            "output_dir",
            "finished_at",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = _utc_now()
        columns = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [task_id]
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE tasks SET {columns} WHERE id = ?", values)
            conn.commit()

    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row) if row else None

    def delete_task(self, task_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()

    def save_draft(self, name: str, config: dict[str, Any]) -> str:
        draft_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO drafts(id, name, config_json, updated_at) VALUES (?, ?, ?, ?)",
                (draft_id, name, json.dumps(config, ensure_ascii=False), now),
            )
            conn.commit()
        return draft_id

    def list_drafts(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM drafts ORDER BY updated_at DESC").fetchall()
            return [dict(row) for row in rows]

    def register_dataset(self, name: str, yaml_path: str, root_path: str = "", class_count: int = 0) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets(id, name, yaml_path, root_path, class_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    yaml_path = excluded.yaml_path,
                    root_path = excluded.root_path,
                    class_count = excluded.class_count
                """,
                (dataset_id, name, yaml_path, root_path, class_count, now),
            )
            conn.commit()
        return dataset_id

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def delete_dataset(self, name: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
            conn.commit()
