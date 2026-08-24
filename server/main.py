from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

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


def _resolve_dir(value: str, default_rel: str) -> Path:
    """将设置中的目录解析为绝对路径：绝对路径直接用，相对路径基于项目根；空值用默认。"""
    value = (value or "").strip()
    if not value:
        return (BASE_DIR / default_rel).resolve()
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


dataset_service = DatasetService(
    BASE_DIR,
    db,
    datasets_dir=_resolve_dir(db.get_setting("datasets_cfg_dir", ""), "datasets"),
)


def _is_within(base: Path, target: Path) -> bool:
    """严格判断 target 是否位于 base 目录之内（防止 startswith 前缀绕过的路径穿越）。"""
    try:
        return target.resolve().is_relative_to(base.resolve())
    except (ValueError, OSError):
        return False


# 图片格式识别（用于 JSON 标注转 YOLO 时获取归一化所需的宽高，无需 Pillow）
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _parse_image_header(header: bytes) -> tuple[int, int] | None:
    """从图片文件头字节解析 (宽, 高)，识别 JPEG / PNG / BMP / WebP。"""
    if header[:2] == b"\xff\xd8":  # JPEG：扫描 SOF 段
        i = 2
        while i < len(header) - 9:
            if header[i] != 0xFF:
                i += 1
                continue
            marker = header[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return struct.unpack(">HH", header[i + 5 : i + 9])
            i += 1
    elif header[:8] == b"\x89PNG\r\n\x1a\n":  # PNG：IHDR 宽高
        return struct.unpack(">II", header[16:24])
    elif header[:2] == b"BM":  # BMP
        return struct.unpack("<II", header[18:26])
    elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":  # WebP
        if header[12:16] == b"VP8 ":
            w, h = struct.unpack("<HH", header[26:30])
            return (w & 0x3FFF, h & 0x3FFF)
        if header[12:16] == b"VP8L":
            b0, b1, b2, b3, b4 = header[21:26]
            return (1 + (((b1 & 0x3F) << 8) | b0), 1 + (((b3 & 0xF) << 10) | (b2 << 2) | (b4 >> 6)))
        if header[12:16] == b"VP8X":
            w = 1 + (struct.unpack("<I", header[24:28])[0] & 0xFFFFFF)
            h = 1 + (struct.unpack("<I", header[28:32])[0] & 0xFFFFFF)
            return (w, h)
    return None


def _read_image_size(path: Path) -> tuple[int, int] | None:
    """解析图片文件头，返回 (宽, 高)，失败返回 None。"""
    try:
        with open(path, "rb") as f:
            header = f.read(1024)
    except OSError:
        return None
    return _parse_image_header(header)


def _size_from_image_data(image_data: str) -> tuple[int, int] | None:
    """从 LabelMe 标注的 imageData（base64 编码的图片）解析 (宽, 高)。"""
    if not image_data:
        return None
    try:
        b64 = image_data.strip()
        prefix = b64[:1400]
        header = base64.b64decode(prefix + "=" * ((4 - len(prefix) % 4) % 4))
        return _parse_image_header(header)
    except Exception:
        return None


def _find_same_image(images_dir: Path, stem: str) -> Path | None:
    """在 images 目录中查找与标注同名的图片文件。"""
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _json_to_yolo_txt(content: bytes, names: list[str], size_hint: tuple[int, int] | None) -> str:
    """将单张图片的 JSON 标注（LabelMe 或常见格式）转换为 YOLO txt 内容。

    - 类别：优先按数据集 yaml 的 names 顺序映射；names 中不存在时回退为数字类别。
    - 宽高：优先取 JSON 内的 imageWidth/imageHeight，否则使用同名图片解析出的尺寸。
    """
    try:
        data = json.loads(content.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise ValueError(f"JSON 解析失败：{exc}")
    if isinstance(data, list):
        shapes, raw = data, {}
    else:
        raw, shapes = data, []
        for key in ("shapes", "annotations", "objects", "regions"):
            if isinstance(data.get(key), list):
                shapes = data[key]
                break
    width = raw.get("imageWidth") or raw.get("width")
    height = raw.get("imageHeight") or raw.get("height")
    try:
        width = int(width) if width else None
        height = int(height) if height else None
    except (TypeError, ValueError):
        width, height = None, None
    if not width or not height:
        # LabelMe 无宽高字段时，优先从 imageData（base64 图片）解析，其次用同名图片
        from_data = _size_from_image_data(raw.get("imageData") or "")
        if from_data:
            width, height = from_data
    if not width or not height:
        if size_hint:
            width, height = size_hint
    if not width or not height:
        raise ValueError("无法确定图片尺寸（JSON 无宽高且未找到同名图片）")
    name_to_id = {str(name): idx for idx, name in enumerate(names)} if names else {}
    lines: list[str] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label") or shape.get("name") or shape.get("class") or ""
        cls = name_to_id.get(str(label))
        if cls is None:
            try:
                cls = int(label)
            except (TypeError, ValueError):
                continue  # 无法确定类别，跳过该框
        points = shape.get("points") or shape.get("box") or []
        xs: list[float] = []
        ys: list[float] = []
        if isinstance(points, list) and points:
            if isinstance(points[0], (list, tuple)):  # 二维点列表（LabelMe 多边形）
                for pt in points:
                    if len(pt) >= 2:
                        try:
                            xs.append(float(pt[0]))
                            ys.append(float(pt[1]))
                        except (TypeError, ValueError):
                            pass
            elif len(points) == 4:  # 扁平 [x1, y1, x2, y2]
                try:
                    xs = [float(points[0]), float(points[2])]
                    ys = [float(points[1]), float(points[3])]
                except (TypeError, ValueError):
                    pass
        if not xs and any(k in shape for k in ("x1", "xmin", "left", "x")):
            try:
                xs = [float(shape.get("x1") or shape.get("xmin") or shape.get("left") or shape.get("x")),
                      float(shape.get("x2") or shape.get("xmax") or shape.get("right") or shape.get("x1") or shape.get("xmin") or shape.get("left") or shape.get("x"))]
                ys = [float(shape.get("y1") or shape.get("ymin") or shape.get("top") or shape.get("y")),
                      float(shape.get("y2") or shape.get("ymax") or shape.get("bottom") or shape.get("y1") or shape.get("ymin") or shape.get("top") or shape.get("y"))]
            except (TypeError, ValueError):
                continue
        if len(xs) < 2 or len(ys) < 2:
            continue
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return "\n".join(lines)

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
    return {"ok": True, "token": token, "username": username, "is_admin": bool(user["is_admin"])}


@app.post("/api/logout")
async def logout(payload: dict[str, str] | None = None) -> dict[str, Any]:
    token = (payload or {}).get("token") or ""
    if token:
        auth_service.revoke_token(token)
    return {"ok": True}


@app.get("/api/auth/me")
async def auth_me(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    username = auth_service.validate_token(authorization[7:].strip())
    if not username:
        raise HTTPException(status_code=401, detail="登录已过期")
    user = db.get_user(username)
    return {"ok": True, "username": username, "is_admin": bool(user["is_admin"]) if user else False}


@app.post("/api/change-password")
async def change_password(payload: dict[str, str], authorization: str | None = Header(default=None)) -> dict[str, Any]:
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


def _current_user(authorization: str | None) -> dict[str, Any] | None:
    """从 Bearer Token 解析当前登录用户。"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    username = auth_service.validate_token(authorization[7:].strip())
    if not username:
        return None
    return db.get_user(username)


# ---------- 用户管理（仅超级管理员） ----------
@app.get("/api/users")
async def users_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    rows = db.get_users()
    return {
        "ok": True,
        "users": [
            {
                "id": r["id"],
                "username": r["username"],
                "is_admin": bool(r["is_admin"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    }


@app.post("/api/users")
async def user_create(payload: dict[str, str], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="仅超级管理员可管理用户")
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username or len(password) < 6:
        raise HTTPException(status_code=400, detail="用户名必填，密码至少 6 位")
    if db.get_user(username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    db.create_user(username, auth_service.hash_password(password), is_admin=False)
    return {"ok": True, "message": f"已新增用户 {username}"}


@app.put("/api/users")
async def user_update(payload: dict[str, str], authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="仅超级管理员可管理用户")
    username = (payload.get("username") or "").strip()
    new_password = payload.get("password") or ""
    target = db.get_user(username)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 位")
    db.update_password(username, auth_service.hash_password(new_password))
    return {"ok": True, "message": f"已重置 {username} 的密码"}


@app.delete("/api/users")
async def user_delete(username: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="仅超级管理员可管理用户")
    target = db.get_user(username)
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")
    if target["is_admin"]:
        raise HTTPException(status_code=400, detail="超级管理员不可删除")
    db.delete_user(username)
    return {"ok": True, "message": f"已删除用户 {username}"}


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
    """列出服务器目录下的子目录，供页面"路径选择"使用。
    相对路径基于项目根目录解析；路径不存在时自动向上定位到最近存在的目录。
    """
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        hint = ""
        if not p.exists():
            cur = p
            while cur != cur.parent and not cur.exists():
                cur = cur.parent
            if cur != p:
                hint = f"路径不存在，已定位到最近目录：{cur}"
                p = cur
        if not p.exists():
            return {"ok": False, "error": "路径不存在"}
        if not p.is_dir():
            p = p.parent
        # 目录为空时不自动向上跳转：保留在目标目录，用户可直接「选择当前目录」确认。
        # （新建数据集时数据根目录可能暂时为空，自动跳转会让人无法选中目标目录）
        try:
            children = sorted(
                (d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")),
                key=str.lower,
            )
        except PermissionError:
            return {"ok": False, "error": "无权限访问该目录"}
        if not children and p != p.parent:
            hint = (hint + "；" if hint else "") + "该目录为空，可直接点击「选择当前目录」确认"
        parent = str(p.parent) if p.parent != p else ""
        return {
            "ok": True,
            "path": str(p),
            "parent": parent,
            "base": str(BASE_DIR),
            "dirs": children,
            "hint": hint,
        }
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
    try:
        result = dataset_service.delete_yaml(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@app.post("/api/datasets/annotations/upload")
async def dataset_annotations_upload(
    dataset: str = Form(...),
    split: str = Form("train"),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """上传标注文件到所选数据集的 labels/<split> 目录。

    - `.txt`：YOLO 格式，直接保存（每张图片对应一个同名 .txt）；
    - `.json`：自动转换为 YOLO 格式的 .txt 后保存（图片尺寸优先取 JSON 内的宽高，
      否则读取 images/<split> 中同名图片；类别按数据集 yaml 的 names 顺序映射）。
    - `.zip`：压缩包内的 .txt / .json 均按上述规则处理。
    """
    if split not in {"train", "val", "test"}:
        return {"ok": False, "error": "非法分区：仅支持 train / val / test"}
    name_key = dataset if dataset.endswith(".yaml") else dataset + ".yaml"
    root_path, names = "", []
    for item in dataset_service.list_all():
        if item["name"] == name_key:
            root_path = (item.get("root_path") or "").strip()
            names = item.get("names") or []
            break
    if not root_path:
        return {"ok": False, "error": "数据集不存在：" + dataset}
    root = Path(root_path).expanduser()
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()
    if not _is_within(BASE_DIR, root):
        return {"ok": False, "error": "数据集根目录超出项目范围，已拒绝"}
    labels_dir = (root / "labels" / split).resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images" / split

    def _save_annotation(member: str, content: bytes) -> str | None:
        """写入一个标注文件，返回错误信息（成功返回 None）。"""
        member_path = Path(member)
        if member_path.suffix.lower() == ".txt":
            target = (labels_dir / member_path).resolve()
            if not target.is_relative_to(labels_dir):
                return "非法路径：" + member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return None
        if member_path.suffix.lower() == ".json":
            try:
                image = _find_same_image(images_dir, member_path.stem)
                size_hint = _read_image_size(image) if image else None
                txt = _json_to_yolo_txt(content, names, size_hint)
            except ValueError as exc:
                return f"{member}: {exc}"
            target = (labels_dir / f"{member_path.stem}.txt").resolve()
            if not target.is_relative_to(labels_dir):
                return "非法路径：" + member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(txt, encoding="utf-8")
            return None
        return None

    saved = 0
    converted = 0
    skipped = 0
    errors: list[str] = []
    for file in files:
        raw_name = Path(file.filename or "").name
        content = await file.read()
        if raw_name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for member in zf.namelist():
                        member_path = Path(member)
                        # zip slip 防护：拒绝绝对路径与包含 .. 的路径
                        if member_path.is_absolute() or ".." in member_path.parts:
                            return {"ok": False, "error": "压缩包包含非法路径，已拒绝解压"}
                        if member_path.suffix.lower() not in (".txt", ".json"):
                            continue
                        error = _save_annotation(member, zf.read(member))
                        if error:
                            skipped += 1
                            if len(errors) < 5:
                                errors.append(error)
                        else:
                            saved += 1
                            if member_path.suffix.lower() == ".json":
                                converted += 1
            except zipfile.BadZipFile:
                return {"ok": False, "error": "不是有效的 zip 文件"}
        else:
            error = _save_annotation(raw_name, content)
            if error:
                skipped += 1
                if len(errors) < 5:
                    errors.append(error)
            else:
                saved += 1
                if raw_name.lower().endswith(".json"):
                    converted += 1
    rel = str(labels_dir.relative_to(BASE_DIR.resolve())).replace("\\", "/")
    return {"ok": True, "dir": rel, "count": saved, "converted": converted, "skipped": skipped, "errors": errors}


@app.get("/api/models")
async def models_list() -> list[dict[str, Any]]:
    runs_dir = _resolve_dir(db.get_setting("runs_dir", ""), "runs")
    return list_models(BASE_DIR, runs_dirs=[runs_dir], db=db)


@app.get("/api/models/download")
async def model_download(path: str) -> Any:
    base = BASE_DIR.resolve()
    target = (base / path).resolve()
    if not _is_within(base, target):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    if target.is_file():
        return FileResponse(target, filename=target.name)
    # 目录型转换产物（openvino/ncnn 等 _model 目录）打包为 zip 下载
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(target.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(target).as_posix())
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{target.name}.zip"'},
    )


@app.delete("/api/models")
async def model_delete(path: str) -> dict[str, Any]:
    base = BASE_DIR.resolve()
    target = (base / path).resolve()
    if not _is_within(base, target):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    # OpenVINO / NCNN 等目录型转换产物按目录删除
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True}


@app.post("/api/models/convert")
async def model_convert(payload: ConvertStartRequest) -> dict[str, Any]:
    base = BASE_DIR.resolve()
    target = (base / payload.path).resolve()
    if not _is_within(base, target):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在：{payload.path}")
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
async def settings_get() -> dict[str, Any]:
    settings = db.get_settings()
    # 返回解析后的绝对路径，便于前端展示与回填
    settings["runs_dir"] = str(_resolve_dir(settings.get("runs_dir", ""), "runs"))
    settings["datasets_cfg_dir"] = str(_resolve_dir(settings.get("datasets_cfg_dir", ""), "datasets"))
    settings["datasets_data_dir"] = str(_resolve_dir(settings.get("datasets_data_dir", ""), "datasets"))
    return settings


@app.post("/api/settings")
async def settings_update(payload: SettingsUpdateRequest) -> dict[str, Any]:
    global dataset_service
    if payload.max_parallel_jobs is not None:
        db.set_setting("max_parallel_jobs", str(payload.max_parallel_jobs))
    if payload.active_conda_env is not None:
        db.set_setting("active_conda_env", payload.active_conda_env)
    if payload.runs_dir is not None:
        db.set_setting("runs_dir", payload.runs_dir.strip())
    if payload.datasets_cfg_dir is not None:
        db.set_setting("datasets_cfg_dir", payload.datasets_cfg_dir.strip())
        # 重建数据集服务，使新配置目录立即生效
        dataset_service = DatasetService(
            BASE_DIR,
            db,
            datasets_dir=_resolve_dir(payload.datasets_cfg_dir, "datasets"),
        )
    if payload.datasets_data_dir is not None:
        db.set_setting("datasets_data_dir", payload.datasets_data_dir.strip())
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
async def training_stop(task: str = "") -> dict[str, Any]:
    manager.stop(task)
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
            try:
                metrics = get_hardware_metrics()
            except Exception:
                continue  # 单次指标异常不中断推送
            try:
                await websocket.send_json({"type": "hardware", "data": metrics})
            except Exception:
                break  # 连接已断开

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
    # 禁用 HTML 缓存：页面内联 JS/CSS，浏览器若缓存旧 HTML 会一直看到旧界面，
    # 部署新代码后普通刷新即拿到最新页面（无需强刷）
    return FileResponse(
        BASE_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
