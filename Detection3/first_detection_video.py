from imageai.Detection import VideoObjectDetection
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

execution_path = os.getcwd()

def forFrame(frame_number, output_array, output_count, detected_frame):

    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("Returned Objects is : ", type(detected_frame))
    print("------------END OF A FRAME --------------")

    list_of_box_points = []
    for object in output_array:
        list_of_box_points.append(object['box_points'])
    return (list_of_box_points)

def forFull(output_objects_arrays, count_arrays, average_output_count):
    #Perform action on the 3 parameters returned into the function
    print (output_objects_arrays)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed = "flash")

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "A_FashionWalking2-snipsnip.avi"),
                                output_file_path=os.path.join(execution_path, "fashion_detected"),
                                frames_per_second=40, per_frame_function=forFrame, video_complete_function=forFull, return_detected_frame=True, log_progress=True)


# for eachObject in video_path:
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
