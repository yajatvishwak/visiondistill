from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from visiondistill.config import StudentConfig


class BaseStudent(ABC):
    """Abstract base class for all student models."""

    def __init__(self, config: StudentConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load model weights and any processors into memory."""

    @abstractmethod
    def train(self, data_path: str | Path, project: str | Path | None = None) -> Any:
        """Train the student on the given dataset. Returns training results."""

    @abstractmethod
    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        """Run inference on the given source."""

    @abstractmethod
    def export(self, fmt: str = "onnx", **kwargs: Any) -> Any:
        """Export the model to the specified format."""
