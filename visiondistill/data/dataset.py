from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from visiondistill.data.annotator import IMAGE_EXTENSIONS


def build_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: list[str] | None = None,
    val_split: float = 0.2,
    seed: int = 42,
) -> Path:
    """Organise images + labels into a YOLO dataset directory and write data.yaml.

    Layout created::

        output_dir/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/

    Returns the path to ``data.yaml``.
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
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, img_out / img_path.name)
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_path.name)
            else:
                (lbl_out / f"{img_path.stem}.txt").write_text("")

    data: dict[str, Any] = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return data_yaml
