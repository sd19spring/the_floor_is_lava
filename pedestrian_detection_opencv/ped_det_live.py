from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# initialize the webcam
# 0 is the built in camera for most laptops
# 1 should be the auxillary USB camera, if connected
cap = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):
    # Get a frame from the webcam
    ret, frame = cap.read()
    # Resize (optional) to help speed
    #frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    # Part of the original demo combined overlapping boxes
    # It uses "orig" as a copy of "frame"
    orig = frame.copy()

    # detect people in the image
    # winStride describes the size of the sliding window, larger = faster
    (rects, weights) = hog.detectMultiScale(frame, winStride=(1, 1), padding=(8, 8), scale=1.05)

    # draw the original bounding boxes for detected peds
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the resulting frame
    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
