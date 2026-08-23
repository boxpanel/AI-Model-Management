from __future__ import annotations

import platform
import re
import subprocess
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
    "YOLO11": "yolo11n.pt",
    "YOLOv8": "yolov8n.pt",
    "YOLOv5": "yolov5nu.pt",
}

def _cpu_model() -> str:
    """获取 CPU 型号：优先 /proc/cpuinfo 的 model name（Linux 下 platform.processor 常为空）。"""
    name = platform.processor() or ""
    if not name:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return name or platform.machine() or ""


def _cpu_count() -> int:
    """获取服务器逻辑核心数（与硬件监控一致，均为服务端数据）。"""
    try:
        import psutil

        return psutil.cpu_count(logical=True) or 0
    except Exception:
        return 0


def _igpu_model() -> str:
    """获取核显（集成显卡）型号：通过 lspci 解析非 NVIDIA 的 VGA 设备。"""
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0:
            return ""
        for line in result.stdout.splitlines():
            low = line.lower()
            if "vga" not in low or "nvidia" in low:
                continue
            name = line.split("VGA compatible controller:", 1)[-1].strip()
            name = re.sub(r"\s*\(rev[^)]*\)\s*$", "", name)
            bracketed = re.search(r"\[([^\]]+)\]", name)
            if bracketed:
                return bracketed.group(1).strip()
            if name:
                return name
    except Exception:
        pass
    return ""


CPU_MODEL = _cpu_model()
CPU_COUNT = _cpu_count()
IGPU_MODEL = _igpu_model()


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
    probe_reason = ""
    if resolved_env:
        python_exe = get_conda_python(resolved_env)
        if python_exe:
            probe = probe_env(python_exe)
            if probe.get("ready"):
                ultralytics_ok = bool(probe.get("ultralytics_available"))
                cuda_ok = bool(probe.get("cuda_available"))
                gpu_name = probe.get("gpu_name")
                gpu_count = int(probe.get("gpu_count", 0))
                gpu_names = list(probe.get("gpu_names") or ())
                if not gpu_names and gpu_name:
                    gpu_names = [gpu_name]
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
                    cuda_version=str(probe.get("cuda_version") or ""),
                    gpu_name=gpu_name,
                    gpu_names=gpu_names,
                    gpu_count=gpu_count,
                    cpu_model=CPU_MODEL,
                    cpu_count=CPU_COUNT,
                    igpu_model=IGPU_MODEL,
                    message=message,
                    conda_available=conda_available(),
                    conda_env=resolved_env,
                    conda_envs=[env["name"] for env in list_conda_envs()],
                )
            probe_reason = str(probe.get("message", "")).strip()
            if probe_reason:
                probe_reason = f"；环境探测失败（{python_exe}）：{probe_reason}"

    ultralytics_ok = check_ultralytics()
    cuda_ok, gpu_name, gpu_count = check_cuda()
    ready = ultralytics_ok
    message = "当前 Python 环境就绪，可开始训练" if ready else "未安装 ultralytics，请运行 pip install -r requirements.txt"
    if ready and not cuda_ok:
        message = "ultralytics 已就绪；未检测到 CUDA，将使用 CPU 训练"
    if probe_reason:
        message += probe_reason
    elif not conda_available():
        message += "；未检测到 conda（训练将使用当前 Python）"
    elif not resolved_env:
        message += f"；未找到 {name} 对应 Conda 环境，使用当前 Python"
    return EnvironmentResponse(
        name=name,
        ready=ready,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ultralytics_available=ultralytics_ok,
        cuda_available=cuda_ok,
        cuda_version="",
        gpu_name=gpu_name,
        gpu_names=[gpu_name] if gpu_name else [],
        gpu_count=gpu_count,
        cpu_model=CPU_MODEL,
        cpu_count=CPU_COUNT,
        igpu_model=IGPU_MODEL,
        message=message,
        conda_available=conda_available(),
        conda_env=resolved_env or "",
        conda_envs=[env["name"] for env in list_conda_envs()],
    )


def default_weights_for_model(model_version: str) -> str:
    return MODEL_WEIGHTS.get(model_version, "yolo11n.pt")
