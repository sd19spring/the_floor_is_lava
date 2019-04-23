from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "airport.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"),  minimum_percentage_probability=30)

for eachObject in detections:
# print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
# detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "airport.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True)
