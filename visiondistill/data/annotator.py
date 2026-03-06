from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from visiondistill.data.converter import masks_to_label_file
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
) -> int:
    """Run the teacher on every image in *images_dir* and write YOLO labels.

    Returns the number of images that produced at least one mask.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    annotated = 0
    for img_path in tqdm(image_paths, desc="Annotating"):
        image = Image.open(img_path).convert("RGB")
        prompt = _resolve_prompt(img_path.name, prompts)

        try:
            result = teacher.generate_masks(image, prompt)
        except Exception:
            logger.warning("Failed to generate masks for %s", img_path.name, exc_info=True)
            continue

        if result.masks.size == 0:
            logger.debug("No masks produced for %s", img_path.name)
            label_content = ""
        else:
            label_content = masks_to_label_file(result.masks, class_ids=class_ids)
            annotated += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text(label_content)

    logger.info("Annotated %d / %d images", annotated, len(image_paths))
    return annotated


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
