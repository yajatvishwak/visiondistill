"""Minimal CLI entry point for visiondistill."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from visiondistill.config import PipelineConfig, StudentConfig, TeacherConfig
from visiondistill.pipeline import DistillationPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="visiondistill",
        description="Distill foundation vision models into compact student models via pseudo-labeling.",
    )
    p.add_argument("images_dir", type=Path, help="Directory containing input images.")
    p.add_argument("-o", "--output-dir", type=Path, default="./runs/distill", help="Output directory.")

    # Teacher args
    t = p.add_argument_group("teacher")
    t.add_argument("--teacher-model", default="sam3", choices=["sam2", "sam3", "grounding_dino"])
    t.add_argument("--teacher-weights", default=None, help="HuggingFace model id or local path.")
    t.add_argument("--prompt-type", default="text", choices=["auto", "points", "boxes", "text", "image_exemplar"])
    t.add_argument("--prompt", nargs="+", default=None, help="Global text prompt(s) applied to every image (e.g. --prompt car truck).")
    t.add_argument("--prompts-json", type=Path, default=None, help="JSON file mapping filenames to prompts (overrides --prompt).")
    t.add_argument("--threshold", type=float, default=0.5)

    # Student args
    s = p.add_argument_group("student")
    s.add_argument("--task", default=None, choices=["segment", "detect"],
                   help="Task type. Defaults to 'detect' for grounding_dino, 'segment' otherwise.")
    s.add_argument("--student-model", default=None,
                   help="Ultralytics model string or HF checkpoint (e.g. yolov8n.pt, yolov9c.pt, yolo11n.pt). "
                        "Defaults based on task: yolov8n-seg.pt for segment, yolov8n.pt for detect.")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=16)

    # Pipeline args
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--device", default="auto", help="Device: auto, cuda, mps, or cpu.")
    p.add_argument("--class-names", nargs="+", default=None, help="Class names for the dataset.")
    p.add_argument("--skip-annotation", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    task = args.task
    if task is None:
        task = "detect" if args.teacher_model == "grounding_dino" else "segment"

    if args.teacher_model == "grounding_dino" and not args.class_names:
        print(
            "Error: --class-names is required when using grounding_dino "
            "(class names are used as the detection query).",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = None
    if args.prompts_json is not None:
        prompts = json.loads(args.prompts_json.read_text())
    elif args.prompt is not None:
        prompts = args.prompt if len(args.prompt) > 1 else args.prompt[0]

    student_kwargs: dict = {
        "task": task,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    if args.student_model is not None:
        student_kwargs["model"] = args.student_model

    pipeline = DistillationPipeline(
        teacher=TeacherConfig(
            model=args.teacher_model,
            weights=args.teacher_weights,
            prompt_type=args.prompt_type,
            threshold=args.threshold,
        ),
        student=StudentConfig(**student_kwargs),
        config=PipelineConfig(
            output_dir=args.output_dir,
            val_split=args.val_split,
            device=args.device,
        ),
    )

    data_yaml = pipeline.run(
        images_dir=args.images_dir,
        prompts=prompts,
        class_names=args.class_names,
        skip_annotation=args.skip_annotation,
        skip_training=args.skip_training,
    )
    print(f"Done. Dataset: {data_yaml}")


if __name__ == "__main__":
    main()
