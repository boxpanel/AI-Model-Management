from __future__ import annotations

import sys
from typing import Optional

from .conda_env import (
    conda_available,
    get_conda_python,
    list_conda_envs,
    probe_env,
    resolve_env_for_model,
)
from .schemas import EnvironmentResponse


MODEL_WEIGHTS = {
    "YOLOv11": "yolo11n.pt",
    "YOLOv8": "yolov8n.pt",
    "YOLOv5": "yolov5nu.pt",
}


def check_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


def check_cuda() -> tuple[bool, Optional[str], int]:
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0), torch.cuda.device_count()
    except Exception:
        pass
    return False, None, 0


def get_environment(name: str = "YOLOv11", active_conda_env: str = "") -> EnvironmentResponse:
    resolved_env = resolve_env_for_model(name, active_conda_env)
    if resolved_env:
        python_exe = get_conda_python(resolved_env)
        if python_exe:
            probe = probe_env(python_exe)
            if probe.get("ready"):
                ultralytics_ok = bool(probe.get("ultralytics_available"))
                cuda_ok = bool(probe.get("cuda_available"))
                gpu_name = probe.get("gpu_name")
                gpu_count = int(probe.get("gpu_count", 0))
                ready = ultralytics_ok
                message = f"Conda 环境 {resolved_env} 已就绪"
                if ready and not cuda_ok:
                    message = f"Conda 环境 {resolved_env} 已就绪；将使用 CPU 训练"
                if not ready:
                    message = f"Conda 环境 {resolved_env} 缺少 ultralytics"
                return EnvironmentResponse(
                    name=resolved_env,
                    ready=ready,
                    python_version=str(probe.get("python_version", "")),
                    ultralytics_available=ultralytics_ok,
                    cuda_available=cuda_ok,
                    gpu_name=gpu_name,
                    gpu_count=gpu_count,
                    message=message,
                    conda_available=conda_available(),
                    conda_env=resolved_env,
                    conda_envs=[env["name"] for env in list_conda_envs()],
                )

    ultralytics_ok = check_ultralytics()
    cuda_ok, gpu_name, gpu_count = check_cuda()
    ready = ultralytics_ok
    message = "当前 Python 环境就绪，可开始训练" if ready else "未安装 ultralytics，请运行 pip install -r requirements.txt"
    if ready and not cuda_ok:
        message = "ultralytics 已就绪；未检测到 CUDA，将使用 CPU 训练"
    if conda_available() and not resolved_env:
        message += f"；未找到 {name} 对应 Conda 环境，使用当前 Python"
    return EnvironmentResponse(
        name=name,
        ready=ready,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ultralytics_available=ultralytics_ok,
        cuda_available=cuda_ok,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        message=message,
        conda_available=conda_available(),
        conda_env=resolved_env or "",
        conda_envs=[env["name"] for env in list_conda_envs()],
    )


def default_weights_for_model(model_version: str) -> str:
    return MODEL_WEIGHTS.get(model_version, "yolo11n.pt")
