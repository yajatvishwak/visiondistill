from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from visiondistill.config import StudentConfig
from visiondistill.students.base import BaseStudent

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class SegFormerStudent(BaseStudent):
    """Wraps HuggingFace SegFormer for semantic segmentation training."""

    def __init__(self, config: StudentConfig, device: str = "cpu") -> None:
        super().__init__(config, device)
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        id2label = self.config.id2label or {
            0: "background",
            **{i + 1: f"class_{i}" for i in range(self.config.num_labels - 1)},
        }
        label2id = self.config.label2id or {v: k for k, v in id2label.items()}

        self._processor = SegformerImageProcessor(
            do_resize=True,
            size={"height": self.config.imgsz, "width": self.config.imgsz},
            do_normalize=True,
        )
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            self.config.model,
            num_labels=self.config.num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self._model.to(self.device)
        logger.info("Loaded SegFormer from %s (%d labels)", self.config.model, self.config.num_labels)

    def train(self, data_path: str | Path, project: str | Path | None = None) -> Any:
        if self._model is None:
            self.load()

        from datasets import Dataset
        from transformers import Trainer, TrainingArguments

        data_path = Path(data_path)
        train_ds = self._load_split(data_path / "train")
        val_ds = self._load_split(data_path / "val")

        output_dir = str(Path(project) if project else data_path / "output")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch,
            per_device_eval_batch_size=self.config.batch,
            learning_rate=self.config.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            fp16=self.device != "cpu" and torch.cuda.is_available(),
            **self.config.train_kwargs,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self._compute_metrics,
            data_collator=self._collate,
        )

        result = trainer.train()
        trainer.save_model(output_dir)
        self._processor.save_pretrained(output_dir)
        logger.info("SegFormer training complete. Model saved to %s", output_dir)
        return result

    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        image = Image.open(source).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        upsampled = nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        return upsampled.argmax(dim=1).squeeze().cpu().numpy()

    def export(self, fmt: str = "onnx", **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if fmt != "onnx":
            raise ValueError(f"SegFormerStudent currently only supports 'onnx' export, got '{fmt}'")

        export_path = kwargs.get("output_path", "segformer.onnx")
        dummy = torch.randn(1, 3, self.config.imgsz, self.config.imgsz).to(self.device)
        torch.onnx.export(
            self._model,
            (dummy,),
            export_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
        )
        logger.info("Exported SegFormer to %s", export_path)
        return export_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_split(self, split_dir: Path) -> Any:
        from datasets import Dataset

        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"

        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )

        records: list[dict[str, Any]] = []
        for img_path in image_paths:
            mask_path = masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            records.append({"image_path": str(img_path), "mask_path": str(mask_path)})

        ds = Dataset.from_list(records)
        ds = ds.map(self._preprocess, remove_columns=["image_path", "mask_path"])
        ds.set_format("torch")
        return ds

    def _preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(example["image_path"]).convert("RGB")
        mask = np.array(Image.open(example["mask_path"]))

        encoded = self._processor(images=image, return_tensors="np")
        pixel_values = encoded["pixel_values"].squeeze(0)

        mask_resized = np.array(
            Image.fromarray(mask).resize(
                (self.config.imgsz // 4, self.config.imgsz // 4),
                resample=Image.NEAREST,
            )
        )

        return {
            "pixel_values": pixel_values.astype(np.float32),
            "labels": mask_resized.astype(np.int64),
        }

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([torch.as_tensor(b["pixel_values"]) for b in batch])
        labels = torch.stack([torch.as_tensor(b["labels"]) for b in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    def _compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        logits_tensor = torch.as_tensor(logits)
        upsampled = nn.functional.interpolate(
            logits_tensor, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        preds = upsampled.argmax(dim=1).numpy()

        correct = (preds == labels).sum()
        total = labels.size
        pixel_acc = correct / total if total > 0 else 0.0

        ious: list[float] = []
        for cls_id in range(self.config.num_labels):
            pred_mask = preds == cls_id
            true_mask = labels == cls_id
            intersection = (pred_mask & true_mask).sum()
            union = (pred_mask | true_mask).sum()
            if union > 0:
                ious.append(intersection / union)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        return {"pixel_accuracy": pixel_acc, "mean_iou": mean_iou}
