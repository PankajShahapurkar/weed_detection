import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("runs/detect/weed_training/rtx4090_yolo11m5/results.csv")


# Plot mAP curves
plt.figure()
plt.plot(df["metrics/mAP50(B)"])
plt.plot(df["metrics/mAP50-95(B)"])
plt.title("mAP over Epochs")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.legend(["mAP50", "mAP50-95"])
plt.show()

# Plot losses
plt.figure()
plt.plot(df["train/box_loss"])
plt.plot(df["train/cls_loss"])
plt.plot(df["train/dfl_loss"])
plt.title("Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Box", "Cls", "DFL"])
plt.show()