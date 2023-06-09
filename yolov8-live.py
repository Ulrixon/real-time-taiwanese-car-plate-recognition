import cv2
import argparse

from ultralytics import YOLO

# import supervision as sv
import numpy as np

# %%
ZONE_POLYGON = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args


def main():
    img_array = []
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("/Users/ryan/Downloads/002.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")
    ret = True
    while ret:
        ret, frame = cap.read()

        result = model.track(frame, agnostic_nms=True, device="mps")
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
        result_frame = result[0].plot()
        cv2.imshow("yolov8", result_frame)
        img_array.append(result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out = cv2.VideoWriter(
        "project.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, args.webcam_resolution
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    main()
