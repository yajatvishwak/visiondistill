from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from visiondistill.config import PromptType, TeacherConfig
from visiondistill.teachers.base import BaseTeacher, MaskOutput
from visiondistill.utils.device import safe_dtype


class GroundingDINOTeacher(BaseTeacher):
    """Grounding DINO teacher for open-vocabulary object detection.

    Accepts text prompts (class names joined by ". ") and returns bounding
    boxes with confidence scores.  Masks are returned as an empty array so the
    existing ``MaskOutput`` contract is satisfied.
    """

    SUPPORTED_PROMPTS = {PromptType.TEXT}

    def load(self) -> None:
        dtype = safe_dtype(self.config.device, self.config.dtype)
        self.processor = AutoProcessor.from_pretrained(self.config.weights)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.weights, torch_dtype=dtype
        ).to(self.config.device)

    def generate_masks(
        self,
        image: Image.Image,
        prompts: Any | None = None,
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError(
                "GroundingDINOTeacher requires text prompts.  Pass class names "
                "as a string (period-separated) or a list of strings."
            )
        text = self._build_query(prompts)

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.model.device
        )
        if inputs.get("pixel_values") is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=self.model.dtype
            )

        with torch.no_grad():
            outputs = self.model(**inputs)

        w, h = image.size
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.config.threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy pixel coords
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]  # list[str]

        if len(boxes) == 0:
            return MaskOutput(
                masks=np.empty((0, 0, 0), dtype=bool),
                scores=np.empty((0,)),
                boxes=np.empty((0, 4)),
                labels=[],
            )

        return MaskOutput(
            masks=np.empty((0, 0, 0), dtype=bool),
            scores=scores,
            boxes=boxes,
            labels=labels,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(prompts: Any) -> str:
        """Convert user-provided prompts into the period-separated query
        string expected by Grounding DINO."""
        if isinstance(prompts, str):
            if "." in prompts:
                return prompts if prompts.endswith(".") else prompts + "."
            return prompts + "."
        if isinstance(prompts, (list, tuple)):
            return ". ".join(str(p) for p in prompts) + "."
        raise TypeError(f"Unsupported prompt type for GroundingDINO: {type(prompts)}")
