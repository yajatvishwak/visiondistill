from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from visiondistill.config import PromptType, TeacherConfig
from visiondistill.teachers.base import BaseTeacher, MaskOutput

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class SAM3Teacher(BaseTeacher):
    """SAM3 teacher supporting text, image-exemplar, box, and point prompts."""

    SUPPORTED_PROMPTS = {
        PromptType.TEXT,
        PromptType.IMAGE_EXEMPLAR,
        PromptType.BOXES,
        PromptType.POINTS,
    }

    def load(self) -> None:
        dtype = _DTYPE_MAP.get(self.config.dtype, torch.float32)
        self.model = Sam3Model.from_pretrained(
            self.config.weights, torch_dtype=dtype
        ).to(self.config.device)
        self.processor = Sam3Processor.from_pretrained(self.config.weights)

    def generate_masks(
        self,
        image: Image.Image,
        prompts: Any | None = None,
    ) -> MaskOutput:
        if self.config.prompt_type == PromptType.TEXT:
            return self._generate_text(image, prompts)
        elif self.config.prompt_type == PromptType.BOXES:
            return self._generate_boxes(image, prompts)
        elif self.config.prompt_type == PromptType.POINTS:
            return self._generate_points(image, prompts)
        elif self.config.prompt_type == PromptType.IMAGE_EXEMPLAR:
            return self._generate_image_exemplar(image, prompts)
        raise ValueError(f"Unsupported prompt type: {self.config.prompt_type}")

    def _post_process(self, outputs: Any, inputs: Any) -> MaskOutput:
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        if len(results["masks"]) == 0:
            return MaskOutput(
                masks=np.empty((0, 0, 0), dtype=bool),
                scores=np.empty((0,)),
                boxes=np.empty((0, 4)),
            )
        masks = np.stack([m.cpu().numpy().astype(bool) for m in results["masks"]])
        scores = np.array([s.item() for s in results["scores"]])
        boxes = (
            np.stack([b.cpu().numpy() for b in results["boxes"]])
            if "boxes" in results and len(results["boxes"]) > 0
            else None
        )
        labels = results.get("labels", None)
        return MaskOutput(masks=masks, scores=scores, boxes=boxes, labels=labels)

    # ---- prompt-specific generation methods ----

    def _generate_text(
        self, image: Image.Image, prompts: Any | None
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError("Text prompts required (str or list[str]).")
        text = prompts if isinstance(prompts, str) else " ".join(prompts)
        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._post_process(outputs, inputs)

    def _generate_boxes(
        self, image: Image.Image, prompts: dict[str, Any] | None
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError("Box prompts required: {'boxes': [[x1,y1,x2,y2], ...], 'labels': [1, ...]}")
        input_boxes = [prompts["boxes"]]
        input_boxes_labels = [prompts.get("labels", [1] * len(prompts["boxes"]))]
        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._post_process(outputs, inputs)

    def _generate_points(
        self, image: Image.Image, prompts: dict[str, Any] | None
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError("Point prompts required: {'points': [[x,y], ...], 'labels': [1, ...]}")
        input_points = [prompts["points"]]
        input_labels = [prompts.get("labels", [1] * len(prompts["points"]))]
        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_points_labels=input_labels,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._post_process(outputs, inputs)

    def _generate_image_exemplar(
        self, image: Image.Image, prompts: Any | None
    ) -> MaskOutput:
        if prompts is None:
            raise ValueError("Image exemplar prompts required (PIL.Image or list[PIL.Image]).")
        exemplar_images = prompts if isinstance(prompts, list) else [prompts]
        inputs = self.processor(
            images=image,
            exemplar_images=exemplar_images,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._post_process(outputs, inputs)
