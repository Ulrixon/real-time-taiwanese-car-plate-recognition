import cv2
import argparse
import pandas as pd
from ultralytics import YOLO

# import supervision as sv
import numpy as np
from PIL import Image

# from skimage import color, data, restoration
import csv
import os
from os import listdir
from os.path import isfile, join

# import easyocr

# from scipy.signal import convolve2d

# import cv2
# reader = easyocr.Reader(["en"], gpu=True)


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
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

    # result = reader.readtext(gray)
    result = ocr_model(img, hide_conf=True)[0]
    array = result.boxes.xyxy.cpu().numpy()[:, 0]
    order = array.argsort()
    # ranks = order.argsort()
    # print(order)
    text_label = label[result.boxes.cls.cpu().numpy()[order].astype(int)]

    # result_frame = result.plot()
    text = "".join(str(x) for x in text_label)

    # for res in result:
    # if len(result) == 1:
    # text = res[1]

    # if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
    #    text = res[1]
    #     text += res[1] + " "

    return cleanup_text(str(text).replace("|", "I"))
    # return 0


# %%
ZONE_POLYGON = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args


def main(video_path):
    # img_array = []
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 3
    lineType = 2
    model = YOLO(
        "/Users/ryan/Documents/py_ai/real_time_object_detect/Automatic_Number_Plate_Detection_Recognition_YOLOv8-main/ultralytics/yolo/v8/detect/best_bbox.pt"
    )
    ret = True
    os.makedirs(
        os.path.dirname(data_path + r"/test_result/" + file + r".csv"),
        exist_ok=True,
    )
    while ret:
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
            if text == "" or len(text) < 6:
                continue

            # cv2.putText(
            #    result_frame,
            #    text,
            #    (int(xyxy[2]), int(xyxy[3])),
            #    font,
            #    fontScale,
            #    fontColor,
            #    thickness,
            #    lineType,
            # )
            if result[0].boxes.id is None:
                continue

            mylist = [
                result[0].boxes.id.cpu().numpy()[i],
                result[0].boxes.conf.cpu().numpy()[i],
                text,
            ]

            with open(data_path + r"/test_result/" + file + r".csv", "a") as myfile:
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

        # cv2.imshow("yolov8", result_frame)
        # cv2.waitKey(0)

        # closing all open windows
        # cv2.destroyAllWindows()
        # img_array.append(result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # out = cv2.VideoWriter(
    #    "project.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, args.webcam_resolution
    # )
    # for i in range(len(img_array)):
    #    out.write(img_array[i])
    # out.release()
    txt = voting(data_path + r"/test_result/" + file + r".csv")
    with open(data_path + r"/test_result/" + r"result.txt", "a") as f:
        f.write(str(file).replace(".mp4", "") + r" " + str(txt) + "\n")


def voting(voting_csv_path):
    file_path = voting_csv_path

    # counting_list=[]
    # for entry in data_array:

    df = pd.read_csv(file_path, header=None)
    string_list = []
    distinct_values = np.int8(df[0].unique())
    for i in distinct_values:
        result = df[df[0] == i].groupby([0, 2], as_index=False).count()
        string_list.append(result[2][np.argmax(np.array(result[1]))])

    # df[1] = df[1].astype(int)
    # result.plot(x=2, y=1, kind="bar")
    return " ".join(str(x) for x in string_list)


if __name__ == "__main__":
    ocr_model = YOLO(
        "/Users/ryan/Documents/py_ai/real_time_object_detect/Automatic_Number_Plate_Detection_Recognition_YOLOv8-main/ultralytics/yolo/v8/detect/best_s_ocr.pt"
    )
    label = np.array(
        [
            "-",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]
    )

    label_folder = []
    total_size = 0
    data_path = "/Users/ryan/Downloads/data2"

    for root, dirts, files in os.walk(data_path):
        total_size += len(files)
        for dirt in dirts:
            label_folder.append(dirt)
        # total_size = total_size+  len(files)
    print("found", total_size, "files.")
    print("folder:", label_folder)

    for file in sorted(files):
        if file == ".DS_Store":
            continue
        print(file)
        filename = data_path + r"/" + file
        main(filename)
