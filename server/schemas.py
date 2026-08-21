from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TrainingState(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    INTERRUPTED = "interrupted"


class TrainingStartRequest(BaseModel):
    task_name: str = "yolo-exp-001"
    model_version: str = "YOLOv11"
    weights_path: Optional[str] = None
    dataset: str = "coco8.yaml"
    dataset_root: str = "datasets/coco8"
    train_path: str = "images/train"
    val_path: str = "images/val"
    test_path: str = ""
    class_count: int = 80
    class_names: list[str] = Field(default_factory=list)
    download_url: str = ""
    device: str = "0"
    epochs: int = 100
    batch: int = 8
    imgsz: int = 640
    lr: float = 0.01
    optimizer: str = "auto"
    patience: int = 100
    workers: int = 8
    cache: str = "false"
    resume: bool = True
    augmentation: bool = True


class TrainingStatusResponse(BaseModel):
    state: TrainingState
    task_name: Optional[str] = None
    epoch: int = 0
    total_epochs: int = 0
    box_loss: Optional[float] = None
    val_loss: Optional[float] = None
    map50: Optional[float] = None
    progress: float = 0.0
    eta_seconds: Optional[int] = None
    message: Optional[str] = None


class EnvironmentResponse(BaseModel):
    name: str
    ready: bool
    python_version: str
    ultralytics_available: bool
    cuda_available: bool
    cuda_version: str = ""
    gpu_name: Optional[str] = None
    gpu_names: list[str] = Field(default_factory=list)
    gpu_count: int = 0
    cpu_model: str = ""
    message: str = ""
    conda_available: bool = False
    conda_env: str = ""
    conda_envs: list[str] = Field(default_factory=list)


class DatasetCreateRequest(BaseModel):
    name: str
    root_path: str
    train_path: str = "images/train"
    val_path: str = "images/val"
    test_path: str = ""
    class_count: int = 1
    class_names: list[str] = Field(default_factory=list)
    download_url: str = ""
    # pose（关键点）数据集专用参数，逗号分隔数字，如 kpt_shape="17, 3"、flip_idx="0, 1, 5"
    kpt_shape: str = ""
    flip_idx: str = ""


class SettingsUpdateRequest(BaseModel):
    max_parallel_jobs: Optional[int] = None
    active_conda_env: Optional[str] = None
    runs_dir: Optional[str] = None  # 训练输出目录（权重结果）
    datasets_cfg_dir: Optional[str] = None  # 数据集配置文件（yaml）目录
    datasets_data_dir: Optional[str] = None  # 训练数据根目录（images / labels）


class ConvertStartRequest(BaseModel):
    path: str = ""  # 源权重文件（相对项目根目录）
    fmt: str = "onnx"  # onnx / engine / tflite / torchscript / openvino / ncnn / rknn
    imgsz: int = 640
    device: str = "cpu"
    rknn_platform: str = "rk3588"  # RKNN 目标平台，如 rk3588 / rk3576 / rk3568 / rk3399pro


class DraftSaveRequest(BaseModel):
    name: str
    config: dict = Field(default_factory=dict)


class HardwareMetrics(BaseModel):
    cpu: dict = Field(default_factory=dict)
    memory: dict = Field(default_factory=dict)
    gpu: list[dict] = Field(default_factory=list)
