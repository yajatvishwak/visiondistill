from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, pipeline as hf_pipeline

from visiondistill.config import PromptType, TeacherConfig
from visiondistill.teachers.base import BaseTeacher, MaskOutput
from visiondistill.utils.device import safe_dtype


class SAM2Teacher(BaseTeacher):
    """SAM2 teacher supporting auto, point, and box prompts."""

    SUPPORTED_PROMPTS = {PromptType.AUTO, PromptType.POINTS, PromptType.BOXES}

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__(config)
        self._auto_pipeline: Any = None

    def load(self) -> None:
        dtype = safe_dtype(self.config.device, self.config.dtype)
        if self.config.prompt_type == PromptType.AUTO:
            self._auto_pipeline = hf_pipeline(
                "mask-generation",
                model=self.config.weights,
                device=self.config.device,
                torch_dtype=dtype,
            )
        else:
            self.model = Sam2Model.from_pretrained(
                self.config.weights, torch_dtype=dtype
            ).to(self.config.device)
            self.processor = Sam2Processor.from_pretrained(self.config.weights)

    def generate_masks(
        self,
        image: Image.Image,
        prompts: Any | None = None,
    ) -> MaskOutput:
        if self.config.prompt_type == PromptType.AUTO:
            return self._generate_auto(image)
        elif self.config.prompt_type == PromptType.POINTS:
            return self._generate_points(image, prompts)
        elif self.config.prompt_type == PromptType.BOXES:
            return self._generate_boxes(image, prompts)
        raise ValueError(f"Unsupported prompt type: {self.config.prompt_type}")

    def _generate_auto(self, image: Image.Image) -> MaskOutput:
        assert self._auto_pipeline is not None
        result = self._auto_pipeline(image, points_per_batch=64)
        masks = np.array([np.array(m) for m in result["masks"]])
        scores = np.array(result.get("scores", np.ones(len(masks))))
        return MaskOutput(masks=masks, scores=scores)

    def _generate_points(
        self, image: Image.Image, prompts: dict[str, Any] | None
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError("Point prompts required: {'points': [[x,y,...]], 'labels': [[1,...]]}")
        input_points = prompts["points"]
        input_labels = prompts["labels"]

        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        # pred_masks shape: (num_objects, num_multimask, H, W) — take best per object
        masks = pred_masks[:, 0].numpy().astype(bool)
        scores_t = outputs.iou_scores.cpu()
        scores = scores_t[:, 0].numpy() if scores_t.ndim >= 2 else scores_t.numpy()
        return MaskOutput(masks=masks, scores=scores)

    def _generate_boxes(
        self, image: Image.Image, prompts: dict[str, Any] | None
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError("Box prompts required: {'boxes': [[[x1,y1,x2,y2]]]}")
        input_boxes = prompts["boxes"]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        masks = pred_masks[:, 0].numpy().astype(bool)
        scores_t = outputs.iou_scores.cpu()
        scores = scores_t[:, 0].numpy() if scores_t.ndim >= 2 else scores_t.numpy()
        boxes = np.array(input_boxes[0]) if input_boxes else None
        return MaskOutput(masks=masks, scores=scores, boxes=boxes)

    def unload(self) -> None:
        super().unload()
        if self._auto_pipeline is not None:
            del self._auto_pipeline
            self._auto_pipeline = None
