from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .auth import auth_service, ensure_default_user
from .database import Database
from .datasets import DatasetService
from .environment import get_environment
from .hardware import get_hardware_metrics
from .models_convert import ConvertManager
from .models_repo import list_models
from .schemas import (
    ConvertStartRequest,
    DatasetCreateRequest,
    DraftSaveRequest,
    EnvironmentResponse,
    SettingsUpdateRequest,
    TrainingStartRequest,
)
from .training import TrainingManager

BASE_DIR = Path(__file__).resolve().parent.parent
db = Database(BASE_DIR / "visionlab.db")
manager = TrainingManager(BASE_DIR, db)
converter = ConvertManager(BASE_DIR)
dataset_service = DatasetService(BASE_DIR, db)


def _is_within(base: Path, target: Path) -> bool:
    """严格判断 target 是否位于 base 目录之内（防止 startswith 前缀绕过的路径穿越）。"""
    try:
        return target.resolve().is_relative_to(base.resolve())
    except (ValueError, OSError):
        return False

app = FastAPI(title="YOLO Training Manager", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    manager.set_event_loop(asyncio.get_running_loop())
    manager.recover()
    # 确保默认管理员账号存在（install.sh 或首次启动创建）
    created_pwd = ensure_default_user(db)
    if created_pwd:
        print(f"[VisionLab] 已创建默认账号 admin / {created_pwd}（请登录后立即修改密码）")


# ---------- 登录认证 ----------
@app.post("/api/login")
async def login(payload: dict[str, str]) -> dict[str, Any]:
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    user = db.get_user(username)
    if not user or not auth_service.verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = auth_service.issue_token(username)
    return {"ok": True, "token": token, "username": username}


@app.post("/api/logout")
async def logout(payload: dict[str, str] | None = None) -> dict[str, Any]:
    token = (payload or {}).get("token") or ""
    if token:
        auth_service.revoke_token(token)
    return {"ok": True}


@app.get("/api/auth/me")
async def auth_me(authorization: str | None = None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    username = auth_service.validate_token(authorization[7:].strip())
    if not username:
        raise HTTPException(status_code=401, detail="登录已过期")
    return {"ok": True, "username": username}


@app.post("/api/change-password")
async def change_password(payload: dict[str, str], authorization: str | None = None) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    username = auth_service.validate_token(authorization[7:].strip())
    if not username:
        raise HTTPException(status_code=401, detail="登录已过期")
    old = payload.get("old_password") or ""
    new = payload.get("new_password") or ""
    user = db.get_user(username)
    if not user or not auth_service.verify_password(old, user["password_hash"]):
        raise HTTPException(status_code=400, detail="原密码错误")
    if len(new) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 位")
    db.update_password(username, auth_service.hash_password(new))
    return {"ok": True, "message": "密码已修改"}


# 业务接口鉴权：除登录/健康检查外，所有 /api/ 请求都需要 Bearer Token
@app.middleware("http")
async def auth_middleware(request: Any, call_next: Any) -> Any:
    path = request.url.path
    if path.startswith("/api/") and path not in ("/api/login", "/api/health"):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "未登录"})
        if not auth_service.validate_token(auth_header[7:].strip()):
            return JSONResponse(status_code=401, content={"detail": "登录已过期"})
    return await call_next(request)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/environment", response_model=EnvironmentResponse)
async def environment(model: str = "YOLOv11") -> EnvironmentResponse:
    return get_environment(model, db.get_setting("active_conda_env"))


@app.post("/api/environment/activate", response_model=EnvironmentResponse)
async def environment_activate(model: str = "YOLOv11") -> EnvironmentResponse:
    """选择模型版本时，激活对应的 Conda 训练环境并返回环境状态。"""
    from .conda_env import resolve_env_for_model

    env_name = resolve_env_for_model(model, "")
    if env_name:
        db.set_setting("active_conda_env", env_name)
    return get_environment(model, env_name or "")


@app.get("/api/hardware")
async def hardware() -> dict[str, Any]:
    return get_hardware_metrics()


@app.get("/api/fs/dirs")
async def fs_dirs(path: str = "/") -> dict[str, Any]:
    """列出服务器目录下的子目录，供页面"路径选择"使用（仅目录，跳过隐藏项）。"""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return {"ok": False, "error": "路径不存在"}
        if not p.is_dir():
            p = p.parent
        children = sorted(
            (d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")),
            key=str.lower,
        )
        parent = str(p.parent) if p.parent != p else ""
        return {"ok": True, "path": str(p), "parent": parent, "dirs": children}
    except PermissionError:
        return {"ok": False, "error": "无权限访问该目录"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/datasets")
async def datasets_list() -> list[dict[str, Any]]:
    return dataset_service.list_all()


@app.post("/api/datasets")
async def dataset_create(payload: DatasetCreateRequest) -> dict[str, Any]:
    try:
        result = dataset_service.create_yaml(
            name=payload.name,
            root_path=payload.root_path,
            train_path=payload.train_path,
            val_path=payload.val_path,
            class_count=payload.class_count,
            class_names=payload.class_names,
            test_path=payload.test_path,
            download_url=payload.download_url,
            kpt_shape=payload.kpt_shape,
            flip_idx=payload.flip_idx,
        )
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.put("/api/datasets")
async def dataset_update(payload: DatasetCreateRequest) -> dict[str, Any]:
    try:
        result = dataset_service.create_yaml(
            name=payload.name,
            root_path=payload.root_path,
            train_path=payload.train_path,
            val_path=payload.val_path,
            class_count=payload.class_count,
            class_names=payload.class_names,
            test_path=payload.test_path,
            download_url=payload.download_url,
            kpt_shape=payload.kpt_shape,
            flip_idx=payload.flip_idx,
        )
        return {"ok": True, **result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.delete("/api/datasets")
async def dataset_delete(name: str) -> dict[str, Any]:
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    if "/" in name or "\\" in name or name in {"..", "."}:
        raise HTTPException(status_code=400, detail="非法名称")
    result = dataset_service.delete_yaml(name)
    return {"ok": True, **result}


@app.post("/api/datasets/upload")
async def dataset_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传数据集压缩包（.zip），解压到 datasets/ 目录下。"""
    raw_name = Path(file.filename or "dataset.zip").name
    if Path(raw_name).suffix.lower() != ".zip":
        return {"error": "仅支持 .zip 压缩包"}
    content = await file.read()
    if len(content) > 5 * 1024 * 1024 * 1024:
        return {"error": "压缩包不能超过 5 GB"}
    base = raw_name[:-4].strip() or "dataset"
    datasets_dir = dataset_service.datasets_dir.resolve()
    extract_to = (datasets_dir / base).resolve()
    if not extract_to.is_relative_to(datasets_dir):
        return {"error": "非法压缩包名称"}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                member_path = Path(member)
                # zip slip 防护：拒绝绝对路径与包含 .. 的路径
                if member_path.is_absolute() or ".." in member_path.parts:
                    return {"error": "压缩包包含非法路径，已拒绝解压"}
                target = (extract_to / member_path).resolve()
                if not target.is_relative_to(extract_to):
                    return {"error": "压缩包包含非法路径，已拒绝解压"}
            extract_to.mkdir(parents=True, exist_ok=True)
            zf.extractall(extract_to)
    except zipfile.BadZipFile:
        return {"error": "不是有效的 zip 文件"}
    rel = str(extract_to.relative_to(BASE_DIR)).replace("\\", "/")
    return {"ok": True, "dir": rel}


@app.get("/api/models")
async def models_list() -> list[dict[str, Any]]:
    return list_models(BASE_DIR)


@app.get("/api/models/download")
async def model_download(path: str) -> FileResponse:
    base = BASE_DIR.resolve()
    target = (base / path).resolve()
    if not _is_within(base, target):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, filename=target.name)


@app.delete("/api/models")
async def model_delete(path: str) -> dict[str, Any]:
    base = BASE_DIR.resolve()
    target = (base / path).resolve()
    if not _is_within(base, target):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    target.unlink()
    return {"ok": True}


@app.post("/api/models/convert")
async def model_convert(payload: ConvertStartRequest) -> dict[str, Any]:
    base = BASE_DIR.resolve()
    target = (base / payload.path).resolve()
    if not _is_within(base, target):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        return converter.start(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/convert")
async def model_convert_status(job: str) -> dict[str, Any]:
    return converter.status(job)


@app.post("/api/models/convert/stop")
async def model_convert_stop(job: str) -> dict[str, Any]:
    return converter.stop(job)


@app.get("/api/drafts")
async def drafts_list() -> list[dict[str, Any]]:
    return db.list_drafts()


@app.post("/api/drafts")
async def draft_save(payload: DraftSaveRequest) -> dict[str, Any]:
    draft_id = db.save_draft(payload.name, payload.config)
    return {"ok": True, "id": draft_id}


@app.get("/api/tasks")
async def tasks_list(limit: int = 50) -> list[dict[str, Any]]:
    return db.list_tasks(limit)


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str) -> dict[str, Any]:
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.delete("/api/tasks/{task_id}")
async def task_delete(task_id: str) -> dict[str, Any]:
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    db.delete_task(task_id)
    return {"ok": True}


@app.get("/api/settings")
async def settings_get() -> dict[str, str]:
    return db.get_settings()


@app.post("/api/settings")
async def settings_update(payload: SettingsUpdateRequest) -> dict[str, Any]:
    if payload.max_parallel_jobs is not None:
        db.set_setting("max_parallel_jobs", str(payload.max_parallel_jobs))
    if payload.active_conda_env is not None:
        db.set_setting("active_conda_env", payload.active_conda_env)
    return {"ok": True, "settings": db.get_settings()}


@app.get("/api/training/status")
async def training_status() -> dict[str, Any]:
    return manager.get_status()


@app.post("/api/training/start")
async def training_start(config: TrainingStartRequest) -> dict[str, Any]:
    try:
        manager.start(config)
        return {"ok": True, "status": manager.get_status()}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/training/stop")
async def training_stop() -> dict[str, Any]:
    manager.stop()
    return {"ok": True, "status": manager.get_status()}


@app.post("/api/upload/weights")
async def upload_weights(file: UploadFile = File(...)) -> dict[str, str]:
    # 文件名消毒：只取文件名部分，丢弃任何目录信息，防止路径穿越
    name = Path(file.filename or "weights.pt").name
    suffix = Path(name).suffix.lower()
    if suffix not in {".pt", ".onnx"}:
        return {"error": "仅支持 .pt 或 .onnx 文件"}
    uploads_dir = manager.uploads_dir.resolve()
    target = (uploads_dir / name).resolve()
    if not _is_within(uploads_dir, target):
        return {"error": "非法文件名"}
    content = await file.read()
    if len(content) > 2 * 1024 * 1024 * 1024:
        return {"error": "文件大小不能超过 2 GB"}
    target.write_bytes(content)
    return {"path": str(target.relative_to(BASE_DIR)).replace("\\", "/"), "filename": target.name}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def listener(payload: dict[str, Any]) -> None:
        loop = manager._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(queue.put_nowait, payload)

    manager.subscribe(listener)
    await websocket.send_json({"type": "hardware", "data": get_hardware_metrics()})
    await websocket.send_json({"type": "status", "data": manager.get_status()})
    for entry in manager.get_recent_logs():
        await websocket.send_json({"type": "log", "message": entry.get("message", ""), "level": entry.get("level", "info")})

    async def pump_events() -> None:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)

    async def pump_hardware() -> None:
        while True:
            await asyncio.sleep(2)
            await websocket.send_json({"type": "hardware", "data": get_hardware_metrics()})

    tasks = [
        asyncio.create_task(pump_events()),
        asyncio.create_task(pump_hardware()),
    ]
    try:
        while True:
            raw = await websocket.receive_text()
            if raw == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        if listener in manager._listeners:
            manager._listeners.remove(listener)
        for task in tasks:
            task.cancel()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")
