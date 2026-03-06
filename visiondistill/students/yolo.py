from __future__ import annotations

from pathlib import Path
from typing import Any

from visiondistill.config import StudentConfig, TaskType


class YOLOStudent:
    """Wraps Ultralytics YOLO for training on pseudo-labeled data."""

    def __init__(self, config: StudentConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        from ultralytics import YOLO

        task_map = {TaskType.SEGMENT: "segment", TaskType.DETECT: "detect"}
        self._model = YOLO(self.config.model, task=task_map[self.config.task])

    def train(self, data_yaml: str | Path, project: str | Path | None = None) -> Any:
        if self._model is None:
            self.load()

        train_args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": self.config.epochs,
            "imgsz": self.config.imgsz,
            "batch": self.config.batch,
            "device": self.device,
        }
        if project is not None:
            train_args["project"] = str(project)
        if self.config.augment is not None:
            train_args.update(self.config.augment.to_dict())
        train_args.update(self.config.train_kwargs)

        return self._model.train(**train_args)

    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._model.predict(source=str(source), device=self.device, **kwargs)

    def export(self, fmt: str = "onnx", **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._model.export(format=fmt, **kwargs)
