from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


MODEL_ENV_CANDIDATES = {
    "YOLOv11": ["YOLOv11", "yolov11", "yolo11", "ultralytics"],
    "YOLOv8": ["YOLOv8", "yolov8", "ultralytics"],
    "YOLOv5": ["YOLOv5", "yolov5", "ultralytics"],
}


def _find_conda() -> Optional[str]:
    """在 PATH 与常见安装位置查找 conda 可执行文件（不依赖用户 PATH 配置）。"""
    candidates = [
        shutil.which("conda"),
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/miniconda3/condabin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        os.path.expanduser("~/miniforge3/bin/conda"),
        "/opt/miniconda3/bin/conda",
        "/opt/anaconda3/bin/conda",
        "/usr/local/miniconda3/bin/conda",
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def conda_available() -> bool:
    return _find_conda() is not None


def list_conda_envs() -> list[dict[str, str]]:
    conda_bin = _find_conda()
    if not conda_bin:
        return []
    try:
        result = subprocess.run(
            [conda_bin, "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return []
        payload = json.loads(result.stdout)
        envs = []
        for path in payload.get("envs", []):
            name = Path(path).name
            envs.append({"name": name, "path": path})
        return envs
    except Exception:
        return []


def resolve_env_for_model(model_version: str, preferred: str = "") -> Optional[str]:
    if preferred:
        return preferred
    envs = list_conda_envs()
    prefix = _current_conda_prefix()
    env_names = {env["name"].lower(): env["name"] for env in envs}
    for candidate in MODEL_ENV_CANDIDATES.get(model_version, [model_version]):
        cand_lower = candidate.lower()
        matched = env_names.get(cand_lower)
        if not matched:
            continue
        # 同名环境优先选择当前 conda（/root/miniconda3 等）前缀下的，避免匹配到其他用户的环境
        for env in envs:
            if env["name"] == matched and prefix and env["path"].startswith(prefix):
                return matched
        return matched
    return None


def _current_conda_prefix() -> str:
    """返回当前使用的 conda 根目录（如 /root/miniconda3），用于优先匹配同源环境。"""
    conda_bin = _find_conda()
    if not conda_bin:
        return ""
    return str(Path(conda_bin).resolve().parent.parent)


def get_conda_python(env_name: str) -> Optional[str]:
    conda_bin = _find_conda()
    if not conda_bin or not env_name:
        return None
    envs = list_conda_envs()
    prefix = str(Path(conda_bin).resolve().parent.parent)
    # 优先当前 conda 前缀下的同名环境
    for env in envs:
        if env["name"] == env_name and env["path"].startswith(prefix):
            candidate = Path(env["path"]) / "bin" / "python"
            return str(candidate) if candidate.exists() else None
    target = next((env for env in envs if env["name"] == env_name), None)
    if not target:
        return None
    candidate = Path(target["path"]) / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def probe_env(python_executable: str) -> dict[str, Any]:
    # 注意：ultralytics 首次导入会向 stdout 打印设置初始化欢迎信息（Creating new
    # Ultralytics Settings...），必须用 redirect 抑制，否则会污染下方的 JSON 输出导致解析失败
    script = (
        "import json, os, sys\n"
        "import contextlib\n"
        "_sink = open(os.devnull, 'w')\n"
        "data = {'python_version': '.'.join(map(str, sys.version_info[:3])), "
        "'ultralytics_available': False, 'cuda_available': False, 'cuda_version': None, "
        "'gpu_name': None, 'gpu_names': [], 'gpu_count': 0}\n"
        "with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):\n"
        "    try:\n"
        "        import ultralytics\n"
        "        data['ultralytics_available'] = True\n"
        "    except ImportError:\n"
        "        pass\n"
        "    try:\n"
        "        import torch\n"
        "        data['cuda_available'] = torch.cuda.is_available()\n"
        "        if data['cuda_available']:\n"
        "            data['cuda_version'] = torch.version.cuda\n"
        "            data['gpu_count'] = torch.cuda.device_count()\n"
        "            data['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(data['gpu_count'])]\n"
        "            data['gpu_name'] = data['gpu_names'][0] if data['gpu_names'] else None\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps(data))"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return {"ready": False, "message": result.stderr.strip() or "环境探测失败"}
        try:
            return {"ready": True, **json.loads(result.stdout.strip())}
        except Exception as exc:
            return {"ready": False, "message": f"{exc}；stdout: {result.stdout[:200]!r}"}
    except Exception as exc:
        return {"ready": False, "message": str(exc)}


def create_conda_env(env_name: str) -> tuple[bool, str]:
    conda_bin = _find_conda()
    if not conda_bin:
        return False, "未检测到 conda 命令"
    try:
        result = subprocess.run(
            [conda_bin, "create", "-y", "-n", env_name, "python=3.11"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or "创建环境失败"
        install = subprocess.run(
            [conda_bin, "run", "-n", env_name, "python", "-m", "pip", "install", "ultralytics"],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if install.returncode != 0:
            return False, install.stderr.strip() or "安装 ultralytics 失败"
        return True, f"已创建并配置环境 {env_name}"
    except Exception as exc:
        return False, str(exc)
