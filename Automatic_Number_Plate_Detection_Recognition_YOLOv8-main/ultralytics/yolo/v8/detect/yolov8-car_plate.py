import cv2
import argparse

from ultralytics import YOLO

# import supervision as sv
import numpy as np

import easyocr

# import cv2
reader = easyocr.Reader(["en"], gpu=True)


def ocr_image(img, coordinates):
    x, y, w, h = (
        int(coordinates[0]),
        int(coordinates[1]),
        int(coordinates[2]),
        int(coordinates[3]),
    )
    img = img[y:h, x:w]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    result = reader.readtext(gray)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
    #     text += res[1] + " "

    return str(text)


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

    cap = cv2.VideoCapture("demo.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 3
    lineType = 2
    model = YOLO("best.pt")

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True, device="mps")[0]
        result_frame = result[0].plot()
        # print(result.boxes.xyxy.cpu().numpy())
        for xyxy in result.boxes.xyxy.cpu().numpy():
            # print((int(xyxy[0]), int(xyxy[3])))
            text = ocr_image(frame, xyxy)
            # print(text)
            cv2.putText(
                result_frame,
                text,
                (int(xyxy[2]), int(xyxy[3])),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )
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

        cv2.imshow("yolov8", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
