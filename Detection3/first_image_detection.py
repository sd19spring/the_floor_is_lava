from imageai.Detection import ObjectDetection
import cv2
import os

cap = cv2.VideoCapture(0)
_, frame = cap.read()

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections, detected_objects_image_array = detector.detectObjectsFromImage(input_image=frame, output_image_path='', input_type='array', output_type='array', minimum_percentage_probability=30)

print(type(detections))
print(detected_objects_image_array)

# cv2.imshow('frame', detected_objects_image_array)