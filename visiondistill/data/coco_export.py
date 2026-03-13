"""Accumulates detection results and writes a minimal COCO-format JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class COCOExporter:
    """Incrementally builds a COCO annotations dict and flushes to JSON."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self._images: list[dict[str, Any]] = []
        self._annotations: list[dict[str, Any]] = []
        self._ann_id = 1

    def add_image(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        boxes_xyxy: np.ndarray,
        class_ids: list[int],
        scores: np.ndarray | None = None,
    ) -> None:
        self._images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        for idx, ((x1, y1, x2, y2), cid) in enumerate(zip(boxes_xyxy, class_ids)):
            x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
            ann: dict[str, Any] = {
                "id": self._ann_id,
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            if scores is not None:
                ann["score"] = float(scores[idx])
            self._annotations.append(ann)
            self._ann_id += 1

    def save(self, path: Path) -> Path:
        categories = [
            {"id": i, "name": name} for i, name in enumerate(self.class_names)
        ]
        coco: dict[str, Any] = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(coco, indent=2))
        return path
