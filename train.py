from ultralytics import YOLO
import torch


def main():
    torch.set_float32_matmul_precision('high')

    model = YOLO("runs/detect/weed_training/rtx4090_yolo11m5/weights/last.pt")

    model.train(
        data="weed.yaml",
        epochs=250,
        imgsz=768,
        batch=32,
        device=0,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        patience=35,
        cos_lr=True,
        amp=True,
        pretrained=True,
        workers=8,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project="weed_training",
        name="rtx4090_yolo11m",
        resume=True
    )


if __name__ == "__main__":
    main()