from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from visiondistill.config import PromptType, TeacherConfig


@dataclass
class MaskOutput:
    """Container for a single image's teacher predictions."""

    masks: np.ndarray  # (N, H, W) binary masks
    scores: np.ndarray  # (N,) confidence scores
    boxes: np.ndarray | None = None  # (N, 4) xyxy bounding boxes
    labels: list[str] | None = None  # per-mask class labels (if available)


class BaseTeacher(ABC):
    """Abstract base class for all teacher models."""

    SUPPORTED_PROMPTS: set[PromptType]

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        self._validate_prompt_type()
        self.model: Any = None
        self.processor: Any = None

    def _validate_prompt_type(self) -> None:
        if self.config.prompt_type not in self.SUPPORTED_PROMPTS:
            supported = ", ".join(p.value for p in self.SUPPORTED_PROMPTS)
            raise ValueError(
                f"{self.__class__.__name__} does not support prompt_type="
                f"'{self.config.prompt_type.value}'. Supported: {supported}"
            )

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""

    @abstractmethod
    def generate_masks(
        self,
        image: Image.Image,
        prompts: Any | None = None,
    ) -> MaskOutput:
        """Run inference on a single image and return masks."""

    def unload(self) -> None:
        """Release model from memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
