from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def _read_nvidia_smi() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        gpus = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            name, util, used, total = [part.strip() for part in line.split(",")]
            used_f, total_f, util_f = float(used), float(total), float(util)
            gpus.append(
                {
                    "model": name,
                    "percent": util_f,
                    "memoryUsedGB": round(used_f / 1024, 1),
                    "memoryTotalGB": round(total_f / 1024, 1),
                }
            )
        return gpus
    except Exception:
        return []


def _read_pynvml() -> list[dict[str, Any]]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "model": name,
                    "percent": float(util.gpu),
                    "memoryUsedGB": round(memory.used / (1024**3), 1),
                    "memoryTotalGB": round(memory.total / (1024**3), 1),
                }
            )
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _read_disk() -> dict[str, Any]:
    """读取磁盘分区信息，主分区用于仪表盘展示。"""
    disk = {"percent": 0.0, "usedGB": 0.0, "totalGB": 0.0, "mount": "", "partitions": []}
    try:
        import psutil

        partitions = []
        for part in psutil.disk_partitions(all=False):
            # 跳过光驱/虚拟文件系统等不可用分区
            if not part.fstype or any(k in part.fstype.lower() for k in ("iso9660", "squashfs", "udf")):
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except (OSError, PermissionError):
                continue
            partitions.append(
                {
                    "mount": part.mountpoint,
                    "totalGB": round(usage.total / (1024**3), 1),
                    "usedGB": round(usage.used / (1024**3), 1),
                    "percent": round(usage.percent, 1),
                }
            )
        if partitions:
            main = next((p for p in partitions if p["mount"] in ("/", "C:\\")), partitions[0])
            disk.update(
                {
                    "percent": main["percent"],
                    "usedGB": main["usedGB"],
                    "totalGB": main["totalGB"],
                    "mount": main["mount"],
                    "partitions": partitions,
                }
            )
    except ImportError:
        pass
    return disk


def get_hardware_metrics() -> dict[str, Any]:
    cpu_percent = 0.0
    memory = {"percent": 0.0, "usedGB": 0.0, "totalGB": 0.0}
    cpu_model = platform.processor() or platform.machine()

    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        memory = {
            "percent": vm.percent,
            "usedGB": round(vm.used / (1024**3), 1),
            "totalGB": round(vm.total / (1024**3), 1),
        }
        cpu_model = f"{psutil.cpu_count(logical=True) or '?'} 个逻辑核心"
    except ImportError:
        pass

    gpus = _read_pynvml() or _read_nvidia_smi()
    return {
        "cpu": {"percent": cpu_percent, "model": cpu_model},
        "memory": memory,
        "disk": _read_disk(),
        "gpu": gpus,
    }
