from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from visiondistill.config import TaskType
from visiondistill.data.coco_export import COCOExporter
from visiondistill.data.converter import boxes_to_yolo_label_file, masks_to_label_file
from visiondistill.teachers.base import BaseTeacher

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(images_dir: Path) -> list[Path]:
    """Gather all image files from a directory (non-recursive)."""
    return sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )


Prompts = str | list[str] | dict[str, Any] | None
"""Accepted prompt types:

- ``None``        – no prompts (used with SAM2 auto mode)
- ``str``         – single global text prompt applied to every image
- ``list[str]``   – multiple global text prompts applied to every image
- ``dict``        – per-image mapping ``{filename: prompt_value}``
"""


def annotate_dataset(
    teacher: BaseTeacher,
    images_dir: Path,
    labels_dir: Path,
    prompts: Prompts = None,
    class_ids: list[int] | None = None,
    task: TaskType = TaskType.SEGMENT,
    class_names: list[str] | None = None,
) -> int:
    """Run the teacher on every image and write YOLO labels.

    When *task* is ``DETECT``, writes YOLO-detect bbox labels and a COCO JSON
    annotation file alongside them.  When ``SEGMENT``, writes polygon labels
    (existing behaviour).

    Returns the number of images that produced at least one prediction.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    coco: COCOExporter | None = None
    if task == TaskType.DETECT and class_names:
        coco = COCOExporter(class_names)

    name_to_id: dict[str, int] = {}
    if class_names:
        name_to_id = {n.lower(): i for i, n in enumerate(class_names)}

    annotated = 0
    for img_idx, img_path in enumerate(tqdm(image_paths, desc="Annotating")):
        image = Image.open(img_path).convert("RGB")
        prompt = _resolve_prompt(img_path.name, prompts)

        try:
            result = teacher.generate_masks(image, prompt)
        except Exception:
            logger.warning("Failed to generate predictions for %s", img_path.name, exc_info=True)
            continue

        if task == TaskType.DETECT:
            label_content = _process_detect(
                result, image, img_idx, img_path, class_ids, name_to_id, coco
            )
        else:
            label_content = _process_segment(result, class_ids)

        if label_content:
            annotated += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text(label_content)

    if coco is not None:
        coco_path = labels_dir / "annotations.json"
        coco.save(coco_path)
        logger.info("COCO annotations saved to %s", coco_path)

    logger.info("Annotated %d / %d images", annotated, len(image_paths))
    return annotated


# ---------------------------------------------------------------------------
# Task-specific helpers
# ---------------------------------------------------------------------------


def _process_segment(result: Any, class_ids: list[int] | None) -> str:
    if result.masks.size == 0:
        return ""
    return masks_to_label_file(result.masks, class_ids=class_ids)


def _process_detect(
    result: Any,
    image: Image.Image,
    img_idx: int,
    img_path: Path,
    class_ids: list[int] | None,
    name_to_id: dict[str, int],
    coco: COCOExporter | None,
) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return ""

    w, h = image.size
    resolved_ids = _resolve_class_ids(result, class_ids, name_to_id)

    if coco is not None:
        coco.add_image(
            image_id=img_idx,
            file_name=img_path.name,
            width=w,
            height=h,
            boxes_xyxy=result.boxes,
            class_ids=resolved_ids,
            scores=result.scores,
        )

    return boxes_to_yolo_label_file(
        result.boxes, class_ids=resolved_ids, image_w=w, image_h=h
    )


def _resolve_class_ids(
    result: Any,
    class_ids: list[int] | None,
    name_to_id: dict[str, int],
) -> list[int]:
    """Map teacher-predicted labels to integer class IDs."""
    if class_ids is not None:
        return class_ids

    if result.labels and name_to_id:
        return [name_to_id.get(lbl.lower(), 0) for lbl in result.labels]

    return [0] * len(result.boxes)


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------


def _resolve_prompt(filename: str, prompts: Prompts) -> Any | None:
    """Return the prompt for a given image file.

    - ``None`` → ``None``
    - ``str`` or ``list[str]`` → returned as-is (global prompt for every image)
    - ``dict`` → per-image lookup by *filename*
    """
    if prompts is None:
        return None
    if isinstance(prompts, (str, list)):
        return prompts
    return prompts.get(filename)
