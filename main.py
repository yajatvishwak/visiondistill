"""Example usage of visiondistill."""

from visiondistill import DistillationPipeline, PipelineConfig, StudentConfig, TeacherConfig


def main() -> None:
    pipeline = DistillationPipeline(
        teacher=TeacherConfig(
            model="sam3",
            weights="facebook/sam3",
            prompt_type="text",
        ),
        student=StudentConfig(
            model="yolov8n-seg.pt",
            epochs=100,
            imgsz=640,
        ),
        config=PipelineConfig(
            output_dir="./runs/distill",
            val_split=0.2,
            device="cuda",
        ),
    )

    # Global prompt — applied to every image in the directory
    pipeline.run(
        images_dir="./my_images",
        prompts="car",
        class_names=["car"],
    )

    # Or per-image prompts via dict
    # pipeline.run(
    #     images_dir="./my_images",
    #     prompts={"img1.jpg": ["car", "truck"], "img2.jpg": "person"},
    #     class_names=["car", "truck", "person"],
    # )


if __name__ == "__main__":
    main()
