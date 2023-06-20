# real_time_object_detect

this is a project for detecting car plate, especially tawanese car plate, below has some slides picture briefly tell you my work  
here is a [demo video](https://youtu.be/9aD8Aa4gEUM) of this project  

the code for running this model is in yolov8-double_live.py  
use file search to find it.  
yolov8s_bbox is best_bbox.pt  
yolov8s_ocr is best_s_ocr.pt  

yolov8s_bbox training details are in runs  
yolov8s_ocr training details are in runs_s_ocr file  
including P,PR,R curve and some train and validation pictures


this py file does not contain voting function so if u want use yolov8_test.py  

![1](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/7dc99966-a7be-4ce9-941f-8578100d649f)
![2](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/8ce2f863-9271-4270-9825-ef4d12fbc12f)
all data sources are from roboflow.com  
1.car plat bbox  
source1:[yolov5_LPIv1_DS1](https://universe.roboflow.com/khushal-koli-tcwmx/yolov5_lpiv1-m2q6f/dataset/1)  
source2:[LicensePlates](https://universe.roboflow.com/emil-jahnke/licenseplates-mihfw/dataset/1)  
source3:[regnummer](https://universe.roboflow.com/lemons/regnummer/dataset/11)  
source4:[number_plate](https://universe.roboflow.com/numberplate-qabtg/number_plate-rq8tn/dataset/2)  
2.plate text recognition  
source1:[taiwan-license-plate-char-recognition-research](https://universe.roboflow.com/jackresearch0/taiwan-license-plate-char-recognition-research/dataset/1)  
source2: [carPlate_CHAR](https://universe.roboflow.com/team-m5mtv/carplate_char/dataset/1)  
source3:[check_plates](https://universe.roboflow.com/jogn/check_plates/dataset/1)  
source4:[plate_rec](https://universe.roboflow.com/jogn/plate_rec/dataset/1)  
source5:[license](https://universe.roboflow.com/project-oee82/license-bha52/dataset/7)  
![3](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/5ce634f0-bc75-421f-944a-7b8855d5eeb0)
![4](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/c0b696e7-93cb-4a24-bd3b-2258b6f2dd0a)
![5](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/963a42e6-5a40-4218-9744-04173ecdad0b)
![6](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/c9f6a954-5a95-402b-986e-a67e041db885)
![7](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/4823a453-4bbb-4aff-a15d-e1b08af52f38)

![8](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/26f26c6d-a310-4785-9ee6-eb99b8255d80)
![9](https://github.com/Ulrixon/real_time_object_detect/assets/61776179/f32baed5-cf2a-4553-a525-3c414d84ad73)

# reference
[Automatic Number Plate Detection and Recognition using YOLOv8](https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8)
