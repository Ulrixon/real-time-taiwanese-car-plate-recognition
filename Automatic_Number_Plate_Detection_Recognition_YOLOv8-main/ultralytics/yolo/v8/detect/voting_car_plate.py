# %%
import pandas as pd
import numpy as np

# Specify the file path

file_path = "/Users/ryan/Documents/py_ai/real_time_object_detect/Automatic_Number_Plate_Detection_Recognition_YOLOv8-main/ultralytics/yolo/v8/detect/test.csv"


# counting_list=[]
# for entry in data_array:
# %%

df = pd.read_csv(file_path, header=None)

distinct_values = np.int8(df[0].unique())
for i in distinct_values:
    result = df[df[0] == i].groupby([0, 2], as_index=False).count()
    print(result[2][np.argmax(np.array(result[1]))])

# df[1] = df[1].astype(int)
result.plot(x=2, y=1, kind="bar")

# %%
