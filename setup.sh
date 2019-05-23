#!/usr/bin/env bash
pip install --upgrade -r requirements.txt;
pip uninstall opencv-contrib-python;
pip install opencv-contrib-python;
wget -P source/yolo-coco/ https://pjreddie.com/media/files/yolov3.weights;
