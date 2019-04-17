"""
This file contains the classes used for the project's computer vision.

@author: Duncan Mazza, Elias Gabriel
@revision: v1.1
"""
from skimage.measure import compare_ssim
import cv2
import numpy as np
import imutils


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    n = 0
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = frame.shape
    avg_frame = np.ndarray(shape)
    while n < 30:
        print(n)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_frame = avg_frame + frame
        n += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    avg_frame = avg_frame / 30

    params = cv2.SimpleBlobDetector_Params()
    blob_detector = cv2.SimpleBlobDetector_create(params)
    params.filterByArea = True
    params.minArea = 5
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minConvexity = 0.3
    params.filterByInertia = False
    params.minInertiaRatio = False

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(frame, avg_frame, full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # thresh = cv2.erode(thresh, None, iterations=5)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print(cnts)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 5000:
                continue
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('frame', thresh)
        cv2.imshow('frame2', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
