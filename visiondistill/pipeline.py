from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from visiondistill.config import (
    PipelineConfig,
    StudentConfig,
    TaskType,
    TeacherConfig,
    TeacherModel,
)
from visiondistill.data.annotator import Prompts, annotate_dataset
from visiondistill.data.dataset import build_yolo_dataset
from visiondistill.students.yolo import YOLOStudent
from visiondistill.teachers.base import BaseTeacher

logger = logging.getLogger(__name__)


def _build_teacher(config: TeacherConfig) -> BaseTeacher:
    if config.model == TeacherModel.SAM2:
        from visiondistill.teachers.sam2 import SAM2Teacher
        return SAM2Teacher(config)
    elif config.model == TeacherModel.SAM3:
        from visiondistill.teachers.sam3 import SAM3Teacher
        return SAM3Teacher(config)
    raise ValueError(f"Unknown teacher model: {config.model}")


class DistillationPipeline:
    """End-to-end pseudo-labeling distillation pipeline.

    1. Load the teacher model (SAM2 / SAM3).
    2. Run inference on the user's images to produce segmentation masks.
    3. Convert masks to YOLO-format polygon labels.
    4. Build a YOLO dataset directory with train/val split.
    5. Train a YOLO student model via Ultralytics.
    """

    def __init__(
        self,
        teacher: TeacherConfig | None = None,
        student: StudentConfig | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.teacher_config = teacher or TeacherConfig()
        self.student_config = student or StudentConfig()
        self.config = config or PipelineConfig()

        self.teacher_config.device = self.config.device
        self._teacher: BaseTeacher | None = None
        self._student: YOLOStudent | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        images_dir: str | Path,
        prompts: Prompts = None,
        class_names: list[str] | None = None,
        class_ids: list[int] | None = None,
        skip_annotation: bool = False,
        skip_training: bool = False,
    ) -> Path:
        """Execute the full pipeline. Returns the path to ``data.yaml``."""
        images_dir = Path(images_dir)
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        labels_dir = output_dir / "raw_labels"
        dataset_dir = output_dir / "dataset"

        if not skip_annotation:
            logger.info("Step 1/3: Generating pseudo-labels with %s", self.teacher_config.model.value)
            self._annotate(images_dir, labels_dir, prompts, class_ids)
        else:
            logger.info("Skipping annotation (skip_annotation=True)")

        logger.info("Step 2/3: Building YOLO dataset")
        data_yaml = build_yolo_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=dataset_dir,
            class_names=class_names,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )

        if not skip_training:
            logger.info("Step 3/3: Training student model")
            self._train(data_yaml, project=output_dir / "train")
        else:
            logger.info("Skipping training (skip_training=True)")

        return data_yaml

    def annotate_only(
        self,
        images_dir: str | Path,
        labels_dir: str | Path | None = None,
        prompts: Prompts = None,
        class_ids: list[int] | None = None,
    ) -> Path:
        """Only run the annotation step; returns the labels directory."""
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir) if labels_dir else self.config.output_dir / "raw_labels"
        self._annotate(images_dir, labels_dir, prompts, class_ids)
        return labels_dir

    def train_only(self, data_yaml: str | Path) -> Any:
        """Only run training on an existing YOLO dataset."""
        return self._train(Path(data_yaml), project=self.config.output_dir / "train")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annotate(
        self,
        images_dir: Path,
        labels_dir: Path,
        prompts: Prompts,
        class_ids: list[int] | None,
    ) -> None:
        teacher = self._get_teacher()
        annotate_dataset(
            teacher=teacher,
            images_dir=images_dir,
            labels_dir=labels_dir,
            prompts=prompts,
            class_ids=class_ids,
        )

    def _train(self, data_yaml: Path, project: Path | None = None) -> Any:
        student = self._get_student()
        return student.train(data_yaml, project=project)

    def _get_teacher(self) -> BaseTeacher:
        if self._teacher is None:
            self._teacher = _build_teacher(self.teacher_config)
            self._teacher.load()
        return self._teacher

    def _get_student(self) -> YOLOStudent:
        if self._student is None:
            self._student = YOLOStudent(self.student_config)
            self._student.load()
        return self._student
