"""
Fine-tunes a YOLOv5 detector on the SurgVU tool-detection dataset built by
prepare_tool_detection_data.py (run that first).

Adapted from repo/surgtoolloc's original NVIDIA SurgToolLoc-2022 baseline
(detection_files/run_5fold.sh: yolov5m, 4 GPUs, 5-fold, 300 epochs) down to a single
GPU / time-boxed run via the modern `ultralytics` training API, so no separate
`git clone ultralytics/yolov5 v6.2` checkout is needed. The hyperparameter deviations
the original solution found useful (lr0, rotation, shear, no mixup) are kept.
"""
import argparse
from ultralytics import YOLO


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                         default=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\tool_detection_data\surgvu_tools.yaml")
    parser.add_argument("--model", type=str, default="yolov5s.pt",
                         help="Pretrained checkpoint to fine-tune from (s=small; use yolov5m.pt for more capacity)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str,
                         default=r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\checkpoints\tool_detector")
    parser.add_argument("--name", type=str, default="yolov5s_surgvu")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        optimizer="Adam",
        lr0=5e-4,
        degrees=30.0,
        shear=0.1,
        mixup=0.0,
        patience=20,
    )
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    train()
