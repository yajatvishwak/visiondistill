from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TeacherModel(str, Enum):
    SAM2 = "sam2"
    SAM3 = "sam3"


class PromptType(str, Enum):
    AUTO = "auto"
    POINTS = "points"
    BOXES = "boxes"
    TEXT = "text"
    IMAGE_EXEMPLAR = "image_exemplar"


class StudentModel(str, Enum):
    YOLO = "yolo"
    SEGFORMER = "segformer"


class TaskType(str, Enum):
    SEGMENT = "segment"
    DETECT = "detect"


STUDENT_DEFAULT_WEIGHTS: dict[StudentModel, str] = {
    StudentModel.YOLO: "yolov8n-seg.pt",
    StudentModel.SEGFORMER: "nvidia/mit-b0",
}

DEFAULTS_WEIGHTS: dict[TeacherModel, str] = {
    TeacherModel.SAM2: "facebook/sam2.1-hiera-large",
    TeacherModel.SAM3: "facebook/sam3",
}


@dataclass
class AugmentConfig:
    """Holds optional Ultralytics train() augmentation keyword arguments."""

    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if v is not None}


@dataclass
class TeacherConfig:
    """Configuration for the teacher (foundation) model."""

    model: str | TeacherModel = TeacherModel.SAM3
    weights: str | None = None
    prompt_type: str | PromptType = PromptType.TEXT
    device: str = "auto"
    dtype: str = "float16"
    threshold: float = 0.5
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = TeacherModel(self.model)
        if isinstance(self.prompt_type, str):
            self.prompt_type = PromptType(self.prompt_type)
        if self.weights is None:
            self.weights = DEFAULTS_WEIGHTS[self.model]


@dataclass
class StudentConfig:
    """Configuration for the student model."""

    student_model: str | StudentModel = StudentModel.YOLO
    model: str | None = None
    task: str | TaskType = TaskType.SEGMENT
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    num_labels: int = 2
    id2label: dict[int, str] | None = None
    label2id: dict[str, int] | None = None
    learning_rate: float = 6e-5
    augment: AugmentConfig | dict[str, Any] | None = None
    train_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.student_model, str):
            self.student_model = StudentModel(self.student_model)
        if self.model is None:
            self.model = STUDENT_DEFAULT_WEIGHTS[self.student_model]
        if isinstance(self.task, str):
            self.task = TaskType(self.task)
        if isinstance(self.augment, dict):
            self.augment = AugmentConfig(self.augment)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    output_dir: str | Path = "./runs/distill"
    val_split: float = 0.2
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
