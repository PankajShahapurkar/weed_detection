import cv2
from ultralytics import YOLO

# Load trained model (IMPORTANT: use your trained weights)
model = YOLO("runs/detect/weed_training/rtx4090_yolo11m5/weights/best.pt")

# Input video
video_path = "input.mp4"

# Output video
output_path = "output.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame,device=0)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Write frame to output video
    out.write(annotated_frame)

    # Optional: show live
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output saved as:", output_path)