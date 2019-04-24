from imageai.Detection import VideoObjectDetection
import os
import numpy as np
import cv2
"""
Detects people in input video and saves superimposed boxes video as new file
Requires: opencv, imageai, and Numpy

@author: Gabriella Bourdon, Michael Remley
"""
# Current working directory
execution_path = os.getcwd()

# Set up default camera for capture
camera = cv2.VideoCapture(0)

# Initilize detector
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()

# Locate and load training model from file
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

# Perform detection on video file
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(   execution_path, "traffic-mini.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_mini_detected_1"),
                                frames_per_second=29, log_progress=True)


    # print(video_path)


for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
