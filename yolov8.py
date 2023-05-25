# %%
from ultralytics import YOLO
import torch
import cv2


# %%

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")
results = model(source="https://ultralytics.com/images/bus.jpg", device="mps", conf=0.6)
result_image = results[0]
print(result_image)
# cv2.imshow("mps_test", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# %%
import torch

print(torch.backends.mps.is_available())
# %%
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# do some ops

# %%
