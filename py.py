import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run inference on images
results = model("your_test_images_folder/", save=False)

# Extract confidence scores and labels
y_scores = []
y_true = []

for result in results:
    for box in result.boxes:
        y_scores.append(box.conf.item())  # Confidence score
        y_true.append(1 if box.cls.item() == expected_class else 0)  # 1 for correct detections, 0 otherwise

# Convert to NumPy arrays
y_scores = np.array(y_scores)
y_true = np.array(y_true)

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Plot PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label="YOLOv8 Model")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()
