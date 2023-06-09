from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO(
    "runs/detect/train2/weights/best.pt"
)  # load a pretrained model (recommended for training)
model = YOLO(
    "/Users/ryan/Documents/py_ai/real_time_object_detect/Automatic_Number_Plate_Detection_Recognition_YOLOv8-main/ultralytics/models/v8/yolov8m_car_plate.yaml"
).load(
    "runs/detect/train2/weights/best.pt"
)  # build from YAML and transfer weights


# Train the model
def main():
    model.train(
        data="/Users/ryan/Documents/py_ai/real_time_object_detect/Automatic_Number_Plate_Detection_Recognition_YOLOv8-main/ultralytics/yolo/data/datasets/car_plate.yaml",
        epochs=100,
        imgsz=640,
        device="cpu",
    )


if __name__ == "__main__":
    main()
