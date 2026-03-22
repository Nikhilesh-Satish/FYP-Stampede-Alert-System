import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage fine-tune YOLOv8 for dense crowd person detection."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML (YOLO format).",
    )
    parser.add_argument(
        "--base",
        default="yolov8m.pt",
        help="Base checkpoint (e.g. yolov8m.pt, yolov8l.pt).",
    )
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="stampede_crowd_ft")
    parser.add_argument(
        "--freeze-backbone",
        type=int,
        default=10,
        help="Freeze first N layers during stage 1.",
    )
    return parser.parse_args()


def best_path(project, name):
    return Path(project) / name / "weights" / "best.pt"


def run_validation(weights_path, data_yaml, imgsz):
    best_model = YOLO(str(weights_path))
    best_model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        max_det=1000,
        split="val",
        plots=True,
    )


def main():
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    stage1_name = f"{args.name}_stage1"
    stage2_name = f"{args.name}_stage2"
    model = YOLO(args.base)

    # Stage 1: adapt the detector head quickly on tiles while freezing much of the backbone.
    model.train(
        data=str(data_yaml),
        epochs=args.stage1_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=stage1_name,
        pretrained=True,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.30,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        degrees=3.0,
        translate=0.10,
        scale=0.35,
        shear=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.10,
        max_det=1000,
        patience=15,
        amp=True,
        cache="disk",
        val=True,
        plots=True,
        freeze=args.freeze_backbone,
    )

    stage1_best = best_path(args.project, stage1_name)
    if not stage1_best.exists():
        raise FileNotFoundError(f"Expected best weights not found: {stage1_best}")

    # Stage 2: unfreeze and refine on the same tiles.
    model = YOLO(str(stage1_best))
    model.train(
        data=str(data_yaml),
        epochs=args.stage2_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=stage2_name,
        pretrained=True,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.0004,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        close_mosaic=5,
        mosaic=0.50,
        mixup=0.05,
        copy_paste=0.15,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        degrees=2.0,
        translate=0.08,
        scale=0.25,
        shear=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.05,
        max_det=1000,
        patience=20,
        amp=True,
        cache="disk",
        val=True,
        plots=True,
        resume=False,
    )

    stage2_best = best_path(args.project, stage2_name)
    if not stage2_best.exists():
        raise FileNotFoundError(f"Expected best weights not found: {stage2_best}")

    run_validation(stage2_best, data_yaml, args.imgsz)

    print(f"\nTraining complete. Best weights: {stage2_best}")
    print("Copy to deployment model path:")
    print(f"cp {stage2_best} models/stampede_model.pt")


if __name__ == "__main__":
    main()
