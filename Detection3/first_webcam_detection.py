from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

def parse(pos_num, arr_dicts, key_dict, detected_frame):
    print(pos_num)
    print(arr_dicts)
    print(key_dict)
    # print(detected_frame)
    return None

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="flash", per_frame_function=parse, return_detected_frame=True)




video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_video")
                                , frames_per_second=20, log_progress=True, minimum_percentage_probability=40)
