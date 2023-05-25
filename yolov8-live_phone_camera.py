import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

# %%
ZONE_POLYGON = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True, device="mps")[0]
        # detections = sv.Detections.from_yolov8(result)
        # labels = [
        #    f"{model.model.names[class_id]} {confidence:0.2f}"
        #    for _, confidence, class_id, _ in detections
        # ]
        # frame = box_annotator.annotate(
        #    scene=frame, detections=detections, labels=labels
        # )

        # zone.trigger(detections=detections)
        # frame = zone_annotator.annotate(scene=frame)
        if result is not None:
            cv2.imshow("yolov8", result[0].plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
