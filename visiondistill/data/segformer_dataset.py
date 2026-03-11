from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visiondistill.data.annotator import IMAGE_EXTENSIONS


def _yolo_label_to_mask(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Render a YOLO polygon label file into a per-pixel class-index mask.

    Background pixels are 0; class IDs are stored as ``class_id + 1`` so that
    the mask is compatible with standard semantic segmentation convention where
    0 = background.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask

    text = label_path.read_text().strip()
    if not text:
        return mask

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + at least 3 xy pairs
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]
        points = np.array(
            [[int(x * img_w), int(y * img_h)] for x, y in zip(xs, ys)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], color=class_id + 1)

    return mask


def build_segformer_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: list[str] | None = None,
    val_split: float = 0.2,
    seed: int = 42,
) -> Path:
    """Build a directory of images and PNG semantic masks for SegFormer.

    Layout created::

        output_dir/
        ├── train/
        │   ├── images/
        │   └── masks/
        └── val/
            ├── images/
            └── masks/

    Mask pixel values are class indices (0 = background, 1..N = classes).
    Returns the path to ``output_dir``.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if class_names is None:
        class_names = ["object"]

    random.seed(seed)
    shuffled = list(image_paths)
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - val_split)))
    train_imgs, val_imgs = shuffled[:split_idx], shuffled[split_idx:]

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, img_out / img_path.name)

            img = Image.open(img_path)
            img_w, img_h = img.size

            label_path = labels_dir / f"{img_path.stem}.txt"
            mask = _yolo_label_to_mask(label_path, img_w, img_h)
            mask_img = Image.fromarray(mask, mode="L")
            mask_img.save(mask_out / f"{img_path.stem}.png")

    return output_dir
