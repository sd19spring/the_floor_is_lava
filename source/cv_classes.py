"""
Contains all the classes used for computer vision and backend processing.

@author: Duncan Mazza
@revision: v1.3
"""

import cv2
import cv2.aruco as aruco

from math import acos, cos, sin
import time
import numpy as np
import time
import os
from matplotlib import pyplot as plt

import os
from matplotlib import pyplot as plt

execution_path = os.getcwd()

DETECTION_THRESHOLD = 0.6


def find_square_dest(square):
    """
    Infers what the calibration square would look like if it was orthogonal to the camera. This is set later as the
    destination square: square_dest. The destination square is used for warping the camera perspective to a desireable
    one.

    :param square: list of 4 tuples (coordinates)
    :return: np.float32 list of 4 tuples (coordinates)
    """
    # vector of bottom edge of the calibration square, placed at the origin
    bottom_edge_orig = np.float32([square[2][0] - square[3][0], square[2][1] - square[3][1]])
    mbe = np.linalg.norm(bottom_edge_orig)  # length (magnitude) of the bottom edge (mbe)
    unit_bottom_edge = bottom_edge_orig / mbe  # bottom edge vector changed to length of 1
    theta = abs(acos(unit_bottom_edge.dot(np.float32([1, 0]))))  # absolute angle between horizontal and bottom edge
    # set up the rotation matrix:
    rotation_matrix = np.float32([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    # template for the destination square (to be modified by rotation matrix)
    orig_square_pre = np.float32([[0, 0], [mbe, 0], [mbe, mbe], [0, mbe]])
    # apply the rotation matrix to the original square, translated to match the position of observed calibration square
    return orig_square_pre.dot(rotation_matrix) + np.float32([square[0][0], square[0][1]])


class Stopwatch:
    """
    A utility for the calibrate() method of the ProcessingEngine class; acts as a stopwatch.
    """

    def __init__(self):
        self.logged_time = time.time()
        self.time_since_last_check = 0

    def log_time(self):
        """
        Record the current time
        """
        self.logged_time = time.time()

    def check_stopwatch(self):
        """
        Check the elapsed time between when this method is called and when the stopwatch time was last logged.
        :return: self.time_since_last_check
        """
        self.time_since_last_check = time.time() - self.logged_time
        return self.time_since_last_check


class ByteCapture:
    """
    Serves as a wrapper for a given byte sequence. This class forms a bridge between raw data and
    an abstracted processing engine. Currently it is not used - exists for future capabilities.
    """

    def write(self, byte_seq):
        """ Stores the given byte sequence within an instance attribute. """
        self.bytes = byte_seq

    def read(self):
        """ Returns the stored byte sequence. """
        return None, self.bytes


class Heatmap:
    def __init__(self, cap_dict, num_caps):
        self.heatmap_dict = {}
        for i in range(num_caps):
            cap = cap_dict[i][0]
            _, frame = cap.read()
            camera_shape = frame.shape
            self.heatmap_dict[i] = np.zeros((camera_shape[1], camera_shape[0]))

    def add_heatmap(self, cap_num, box_points):
        self.heatmap_dict[cap_num][box_points[1]:box_points[2], box_points[0]:box_points[3]] += 1

    def show_heatmap(self, cap_num):
        # draw map
        plt.imshow(self.heatmap_dict[cap_num], cmap='hot', interpolation='nearest')
        plt.show()
        plt.savefig('matrix.jpg')


def parse_detected(frame, W, H, layerOutputs, LABELS):
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)

            # TODO: Fix pls
            # if classID != 'person':
            #     continue

            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > DETECTION_THRESHOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, DETECTION_THRESHOLD, 0.3)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = (157, 161, 100)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
    frame_copy = frame
    return boxes, frame_copy


class ProcessingEngine:
    """
    The backend class for image per-processing.
    """

    def __init__(self, threshold=30, debug=False):
        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([os.getcwd(), 'yolo-coco', "yolov3.weights"])
        configPath = os.path.sep.join([os.getcwd(), 'yolo-coco', "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # determine only the *output* layer names that we need from YOLO
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.n = 0  # counter for the calibration process
        self.stopwatch = Stopwatch()
        self.threshold = threshold  # minimum number of frames used to perform calibration
        self.matrix_list = []  # used to store matrices during calibration

        # initialize parameters for ARUCO detection (for perspective correction)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.debug = debug
        self.file_type = False
        self.parameters = aruco.DetectorParameters_create()

        self.record = False  # flag for whether the class should be generating a heatmap or not

        self.cap_dict = {}  # store the OpenCV captures in a dictionary
        self.num_caps = 0  # initialize the value that stores the number of OpenCV captures

        # add all of the OpenCV captures to self.cap_dict:
        i = 0
        while i < 5:  # support up to 5 different cameras
            cap = cv2.VideoCapture(i)

            if cap.isOpened():
                # Representation of dictionary entries: [OpenCV capture, calibration boolean, 0 which is a placeholder
                # for the calibration matrix, and boolean for whether the camera is to be used]
                self.cap_dict[i] = [cap, 0, 0, 1]
            else:  # all the cameras that can be detected have been, so break the loop:
                self.num_caps = i
                break
            i += 1

        self.heatmap = Heatmap(self.cap_dict, self.num_caps)

    def cap_toggle(self, capNum, toggle):
        """
        Switches whether a camera is used for the heatmap building or not
        :param capNum: index of camera being switched on or off
        :param toggle: integer boolean (1 turns it on, 0 turns it off)
        :return: None
        """
        self.cap_dict[capNum][3] = toggle
        return None

    def calib_toggle(self, capNum, toggle):
        """
        Switches whether a camera is being calibrated or not for the heatmap building or not
        :param capNum: index of camera being switched on or off
        :param toggle: integer boolean (1 turns it on, 0 turns it off)
        :return: None
        """

        self.cap_dict[capNum][1] = toggle
        if toggle == 0:  # reset the calibration matrix if the perspective correction is being turned off
            self.cap_dict[capNum][2] = 0
        return None

    def calibrate(self, cap_num):
        """
        Calibrates the camera so that the perspective of the camera will always be orthogonal to the floor or whatever
        surface it is calibrated with. To do this, it looks for the marker calibration sheet. When it finds the markers,
        it uses the four vertices of the observed square, infers what the square would look like if it was in an
        orthogonal perspective, calculates a perspective transformation matrix, and applies that to the frame. It does
        this for self.threshold number of times, logging the transformation matrix each time and returns the average
        transformation matrix. If too much time passes between a frame where the calibration sheet was found and a frame
        where there was not one found, the log of transformation matrices is reset.

        :return: frame or frame converted to bytes, depending on use case
        """
        if self.debug:
            # for visualizing the corners of the markers in debug mode
            colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}

        cap = self.cap_dict[cap_num][0]  # select the camera to be calibrated
        _, frame = cap.read()

        frame_x = frame.shape[1]  # number of pixels wide the camera frame is
        frame_y = frame.shape[0]  # number of pixels tall the camera frame is
        frame_x_c = int(frame_x / 2)  # center of the camera frame in the x direction

        # detect the ARUCO markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if self.stopwatch.check_stopwatch() > 2:
            self.n = 0  # reset the counter if it has been too long; counter is used for ensuring enough frames have
            #             been captured for the average matrix to be calculated
            self.matrix_list = []  # reset the matrix_list if this is the case

        # visualize the process of the calibration process: n frames calibrated / self.threshold needed
        cv2.rectangle(frame, (0, frame_y - 30), (int(self.n * frame_x / self.threshold), frame_y), (157, 161, 100),
                      -1)

        if len(markers) < 4:  # look for one marker sheet
            cv2.putText(frame, "A full calibration sheet isn't visible.", (frame_x_c - int(frame_x / 10), frame_y - 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)  # apply the text
            return frame

        elif len(markers) > 4:  # handle the edge case where multiple calibration sheets are in frame

            cv2.putText(frame, "Whoa that's too many markers! Please only use one calibration sheet at a time.",
                        (frame_x_c - int(frame_x / 6), frame_y - 15), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255, 255, 255), 2)  # apply the text

            return frame

        else:  # one calibration sheet has been detected
            self.n += 1  # update the counter

            # make a dictionary of the markers so that they can be accessed by their id (detection returns them in a
            # random order)
            markers_dict = {}
            for i in range(4):
                markers_dict[ids[i][0]] = markers[i][0]

            # To calculate the required perspective shift, we will need to find how exactly the calibration square
            # skewed. The coordinates of this square as found from a grid of 4 ARUCO markers arranged in a square shape
            # are defined here.
            square = []  # an ordered list that contains the coordinates of the square outlined by the ARUCO markers
            for i in range(4):
                marker = markers_dict[i]
                square.append(tuple(marker[i]))

            # infer what the calibration square should look like if the camera was orthogonal
            square_dest = find_square_dest(square)

            ##### Visualization
            if self.debug:
                # colors that will be used to test whether the order of the stored data is correct
                colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}

                # When in debug mode, the box and its corners will be shown and colorcoded in this order:
                # 1) Top-left (Blue)  2) Top-right (Green)  3) Bottom-right (Red)  4) Bottom left (Cyan)
                for i in range(4):
                    marker = markers_dict[i]
                    # print(markers)
                    if self.debug:
                        for j in range(4):
                            cv2.circle(frame, tuple(marker[j]), 10, colors[j], -1)
                        cv2.putText(frame, "{}".format(i), (int(marker[0][0] + 20), int(marker[0][1] + 20)),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

                # visualize the detected and destination square square_dest (should appear distorted on
                # the perspective-corrected frame):
                for i in range(4):
                    if i is not 3:
                        cv2.line(frame, square[i], square[i + 1], colors[i], 3)
                        cv2.line(frame, tuple(square_dest[i]), tuple(square_dest[i + 1]), colors[i], 3)
                    else:
                        cv2.line(frame, square[i], square[0], colors[i], 3)
                        cv2.line(frame, tuple(square_dest[i]), tuple(square_dest[0]), colors[i], 3)

            else:  # even if not in debug mode, highlight the detected marker square
                for i in range(4):
                    if i is not 3:
                        cv2.line(frame, square[i], square[i + 1], (157, 161, 100), 3)
                    else:
                        cv2.line(frame, square[i], square[0], (157, 161, 100), 3)

            matrix = cv2.getPerspectiveTransform(np.float32(square), square_dest)  # get the transformation matrix
            self.matrix_list.append(matrix)  # record the transformation matrix for this frame
            frame = cv2.warpPerspective(frame, matrix, (frame_x, frame_y))  # apply the transformation matrix

            # take self.threshold frames to calibrate; if calibrated, flip the is_calibrated switch
            if self.n > self.threshold:
                # find the average calibration matrix between the self.threshold frames
                calibration_matrix = np.float32()
                for i in range(self.threshold):  # self.matrix_list should be self.threshold long
                    calibration_matrix = calibration_matrix + self.matrix_list[i]
                self.cap_dict[cap_num][2] = calibration_matrix / self.threshold
                return frame
            else:
                self.stopwatch.log_time()  # log the time
                return frame

    def get_frame(self, cap_num, calibrate=False):
        """
        Returns the captured frame that is warped by the calibration_matrix
        :param: cap_num - index number of the OpenCV capture
        :param: calibrate - boolean for whether the frame should be perspective corrected
        :return: frame or frame converted to bytes, depending on use case
        """
        cap = self.cap_dict.get(cap_num)[0]
        _, frame = cap.read()

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (128, 128),
                                     swapRB=True, crop=False)

        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        boxes, frame = parse_detected(frame, W, H, layerOutputs, self.LABELS)
        print(boxes)

        for i in range(len(boxes)):
            self.heatmap.add_heatmap(cap_num, boxes[i])

        if self.cap_dict[cap_num][1] == 1 or calibrate == True:  # if the camera is in calibration mode:
            if type(self.cap_dict[cap_num][2]) != int:  # already calibrated if true; compare type because when it
                # becomes the calibration matrix, the truth value of a multi-element array is ambiguous
                frame = cv2.warpPerspective(frame, self.cap_dict[cap_num][2], (frame.shape[1], frame.shape[0]))
            else:  # perform calibration
                while self.cap_dict[cap_num][2] == 0:  # not yet calibrated
                    frame = self.calibrate(cap_num)
                    return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

        # if the camera is set to off, then dim the frame by x0.2
        if self.cap_dict[cap_num][3] == 0:
            frame = frame * 0.2

        return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()


if __name__ == "__main__":
    engine = ProcessingEngine(debug=True)

    # display each camera connected to the computer with a corrected perspective
    while True:
        for cap_num in range(engine.num_caps):
            frame = engine.get_frame(cap_num, calibrate=True)
            cv2.imshow("frame {}".format(cap_num), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                engine.heatmap.show_heatmap(cap_num)
                break


