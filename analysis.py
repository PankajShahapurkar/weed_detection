import os
import matplotlib.pyplot as plt

labels_path = "train/labels"

# 1. Class Distribution
class_counts = {}

for file in os.listdir(labels_path):
    with open(os.path.join(labels_path, file)) as f:
        for line in f:
            cls = int(line.split()[0])
            class_counts[cls] = class_counts.get(cls, 0) + 1

plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.title("Class Distribution (Train Set)")
plt.savefig("class_distribution.png")
plt.show()


# 2. Bounding Box Sizes
box_areas = []

for file in os.listdir(labels_path):
    with open(os.path.join(labels_path, file)) as f:
        for line in f:
            _, x, y, w, h = map(float, line.split())
            box_areas.append(w * h)

plt.hist(box_areas, bins=50)
plt.title("Bounding Box Size Distribution")
plt.savefig("bbox_distribution.png")
plt.show()


# 3. Objects per Image
objects_per_image = []

for file in os.listdir(labels_path):
    with open(os.path.join(labels_path, file)) as f:
        objects_per_image.append(len(f.readlines()))

plt.hist(objects_per_image, bins=20)
plt.title("Objects per Image")
plt.savefig("objects_per_image.png")
plt.show()
