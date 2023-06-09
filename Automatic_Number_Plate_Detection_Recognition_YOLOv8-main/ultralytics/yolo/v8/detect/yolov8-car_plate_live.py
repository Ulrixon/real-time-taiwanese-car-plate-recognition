import cv2
import argparse

from ultralytics import YOLO

# import supervision as sv
import numpy as np
from PIL import Image

# from skimage import color, data, restoration
import csv
import easyocr

# from scipy.signal import convolve2d

# import cv2
reader = easyocr.Reader(["en"], gpu=True)


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    # print(text)

    # print(text)
    clean_string = [s for s in text if s.isalnum() or s.isspace()]

    return "".join(clean_string).replace(" ", "")


def set_image_dpi(im):
    length_x, width_y = im.shape
    factor = max(1, float(1024.0 / length_x))
    size1, size2 = int(factor * length_x), int(factor * width_y)
    im_resized = cv2.resize(im, (size2, size1))  # , Image.ANTIALIAS)
    # cv2.imshow("", im_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return im_resized


def to_rectangle(gray):
    contours, hierarchy = cv2.findContours(
        gray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.drawContours(img, contours, 0, (255,255,255), 3)
    # print(contours)
    # print ("contours:",len(contours))
    epsilon = 0.01 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    # print(approx)
    cv2.drawContours(gray, [approx], 0, (125), 3)
    print("simplified contour has", len(approx), "points")
    print("largest contour has ", len(contours[0]), "points")
    cv2.imshow("", gray)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    # new_img=np.zeros(gray.shape)
    # print(approx[0, 0, :])
    print(
        np.array(
            [
                [approx[1, 0, 1], approx[1, 0, 0]],
                [approx[2, 0, 1], approx[2, 0, 0]],
                [approx[0, 0, 1], approx[0, 0, 0]],
                [approx[3, 0, 1], approx[3, 0, 0]],
            ],
            np.float32,
        )
    )
    print(
        np.array(
            [
                [gray.shape[0], 0],
                [gray.shape[1], gray.shape[0]],
                [0, 0],
                [0, gray.shape[1]],
            ],
            np.float32,
        )
    )
    H = cv2.getPerspectiveTransform(
        np.array(
            [
                [approx[1, 0, 1], approx[1, 0, 0]],
                [approx[2, 0, 1], approx[2, 0, 0]],
                [approx[0, 0, 1], approx[0, 0, 0]],
                [approx[3, 0, 1], approx[3, 0, 0]],
            ],
            np.float32,
        ),
        np.array(
            [
                [gray.shape[0], 0],
                [gray.shape[0], gray.shape[1]],
                [0, 0],
                [0, gray.shape[1]],
            ],
            np.float32,
        ),
    )
    gray = cv2.warpPerspective(gray.astype(np.uint8), H, (gray.shape[1], gray.shape[0]))
    cv2.imshow("", gray)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    return gray


def ocr_image(img, coordinates):
    x, y, w, h = (
        int(coordinates[0]),
        int(coordinates[1]),
        int(coordinates[2]),
        int(coordinates[3]),
    )
    img = img[y:h, x:w]
    if img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = set_image_dpi(gray)

    # gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # psf = np.ones((5, 5)) / 25
    # gray = convolve2d(gray, psf, "same")
    # gray += 0.1 * gray.std() * np.random.standard_normal(gray.shape)
    # gray, _ = restoration.unsupervised_wiener(gray, psf)
    # print(gray.shape)
    # gray = cv2.equalizeHist(gray)
    # find contours
    # print(gray.shape)
    # gray = cv2.adaptiveThreshold(    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    result = reader.readtext(gray)
    text = ""

    for res in result:
        # if len(result) == 1:
        text = res[1]

        # if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
        #    text = res[1]
    #     text += res[1] + " "

    return cleanup_text(str(text).replace("|", "I"))


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

    cap = cv2.VideoCapture(
        "/Volumes/xpg_pro512gb/NTUST_EdgeAI-main/yolo_valid/IM_47.jpg"
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 3
    lineType = 2
    model = YOLO(
        "/Users/ryan/Documents/py_ai/real_time_object_detect/Automatic_Number_Plate_Detection_Recognition_YOLOv8-main/ultralytics/yolo/v8/detect/runs/detect/train5/weights/best.pt"
    )

    while True:
        ret, frame = cap.read()

        result = model.track(
            frame,
            agnostic_nms=True,
            conf=0.1,
            device="mps",
            tracker="cartrack.yaml",  # "botsort.yaml",  # stream=True
            persist=True,
        )
        # print(result.boxes.id)
        result_frame = result[0].plot()
        # print(result.boxes.xyxy.cpu().numpy())
        # print(len(result[0]))
        for i in range(len(result[0])):
            if result[0].boxes.xyxy is None:
                continue
            # for xyxy in result.boxes.xyxy.cpu().numpy():
            # print((int(xyxy[0]), int(xyxy[3])))
            xyxy = result[0].boxes.xyxy.cpu().numpy()[i]
            text = ocr_image(frame, xyxy)
            text = text.upper()
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
            if result[0].boxes.id is None:
                continue
            mylist = [
                result[0].boxes.id.cpu().numpy()[i],
                result[0].boxes.conf.cpu().numpy()[i],
                text,
            ]
            with open("test.csv", "a") as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(mylist)
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
        cv2.waitKey(0)

        # closing all open windows
        cv2.destroyAllWindows()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
