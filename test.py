from ultralytics import YOLO
import os


def main():
    # Path to your trained model
    model_path = "weed_training/rtx4090_yolo11m/weights/best.pt"

    # Load trained model
    model = YOLO(model_path)

    # Run prediction on test images folder
    results = model.predict(
        source="weed_dataset/test/images",  # your test images folder
        imgsz=768,
        conf=0.25,          # confidence threshold
        iou=0.5,            # NMS IoU threshold
        device=0,
        save=True,          # save output images
        save_txt=True,      # save predicted labels
        save_conf=True,     # save confidence scores
        project="weed_training",
        name="test_results"
    )

    print("✅ Testing complete. Results saved in:")
    print("weed_training/test_results")


if __name__ == "__main__":
    main()