"""
Contains all the classes used for computer vision and backend processing.
"""

import cv2
import cv2.aruco as aruco
from math import acos, cos, sin
import numpy as np
import time
import datetime
import os

DETECTION_THRESHOLD = 0.6  # minimum confidence level for person to be recognized


class Heatmap:
    """
    This class stores, calculates, and renders the heatmaps generated from the object detection in ProcessingEngine;
    this class also stores previously recorded heatmaps during a given session.

    Each heatmap is a matrix n x m dimensional matrix in the same size of the camera frame (n x m pixels) for which it
    represents. To recognize the presence of a person in the heatmap is to take the bounding boxes detected around that
    person, find the corresponding pixels in the heatmap matrix, and increment them each by 1. The units of the heatmap
    matrices are therefore qualitatively interpreted as density of people observed in a given spot over a given period
    of time.
    """

    def __init__(self):
        self.num_caps = 0  # number of captures (same as ProcessingEngine)
        self.all_heatmaps = {}  # dictionary to store heatmaps across multiple trials
        self.n = -1  # counter for which (sets of) heatmap(s) is currently being recorded to
        self.cap_size_dict = {}  # dictionary to store the resolution of the camera frames

    def reset(self, num_caps, cap_dict):
        """
        Resets the heatmaps using previously stored information about the camera's width and height. It increments
        self.n by 1 and adds the corresponding starter information to the dictionary at the new key self.n + 1.
        :param num_caps: number of cameras connected to the computer
        :cap_dict: dictionary of camera captures from ProcessingEngine
        :return: void
        """
        self.n += 1

        # [{dictionary to store the heatmaps for each camera}, starting time in seconds, ending time in seconds,
        # duration, display starting time, display end time]
        self.all_heatmaps[self.n] = [{}, datetime.datetime.now(), 0, '-', time.ctime(), '-']
        for i in range(num_caps):
            shape = cap_dict[i][4]  # extract the dimensions of the cameras
            self.cap_size_dict[i] = (cap_dict[i][4][0], cap_dict[i][4][1])  # update the capture size dictionary
            self.all_heatmaps[self.n][0][i] = np.zeros(shape)

    def add_to_heatmap(self, cap_num, box_points):
        """
        This method adds the points enclosed by a bounding box to the heatmap. The docstring at the beginning of this
        class provides better context for how this works.
        :param cap_num: index of the camera
        :param box_points: bounding box points (top left x, top left y, bottom right x, bottom right y)
        :return: void
        """
        self.all_heatmaps[self.n][0][cap_num][box_points[1]:box_points[3], box_points[0]:box_points[2]] += 1

    def return_heatmap(self, cap_num, n=-2):
        """
        Returns a matrix that will be interpreted by OpenCV and therefore the user as a heatmap
        :param cap_num:  index of the camera
        :return heatmap_img: a numpy.ndarray that is an image representing the heatmap
        """
        if n == -2:  # -2 is used to indicate the default selection
            n = self.n
        h = self.all_heatmaps[n][0][cap_num]  # retrieve the raw heatmap values
        h = np.interp(h, (0, h.max()), (0, 255))  # rescale the heatmap to 0:255 values
        h = h.astype(np.uint8)  # convert data type for applyColorMap
        heatmap_img = cv2.applyColorMap(h, cv2.COLORMAP_HOT)  # turn matrix into a heatmap
        return heatmap_img

    def record_end_time(self):
        """
        This function records the ending time of a particular recording, calculates the duration of the recording, and
        stores the information accordingly.
        :return: void
        """
        self.all_heatmaps[self.n][2] = datetime.datetime.now()  # record ending time in seconds
        delta = str(self.all_heatmaps[self.n][2] - self.all_heatmaps[self.n][1])  # compute duration of recording
        delta = delta[0:len(delta) - 7]  # round the time
        self.all_heatmaps[self.n][3] = delta
        self.all_heatmaps[self.n][5] = time.ctime()

    def get_time_info(self, n):
        """
        Returns the time information associated with recording n.
        :param n: index of the recording to be returned.
        :return: start time, end time, and duration (in one string)
        """
        return "Start time: " + str(self.all_heatmaps[n][4]) + "\nEnd time: " + str(
            self.all_heatmaps[n][5]) + "\nDuration: " + str(self.all_heatmaps[n][3])


class ProcessingEngine:
    """
    The backend class for image per-processing.
    """

    def __init__(self, threshold=30, debug=False):
        # load the COCO class labels our YOLO model was trained on
        cwd = os.getcwd()
        print(cwd)
        labelsPath = os.path.sep.join(["api/yolo-coco", "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        self.weightsPath = os.path.sep.join([cwd, 'api/yolo-coco', "yolov3.weights"])
        self.configPath = os.path.sep.join([cwd, 'api/yolo-coco', "yolov3.cfg"])
        self.ln = 0  # a placeholder (for determining only the *output* layer names that we need from YOLO)
        # that is overridden when self.turn_on() is called

        self.n = 0  # counter for the calibration process
        self.stopwatch = Stopwatch()
        self.threshold = threshold  # minimum number of frames used to perform calibration
        self.matrix_dict = {}  # used to store matrices during calibration
        # initialize parameters for ARUCO detection (used for perspective correction)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()

        self.record = False  # flag for whether the class should be generating a heatmap or not
        self.debug = debug

        self.cap_dict = {}  # store the OpenCV captures in a dictionary
        self.detect_dict = {}  # store detected outputs in a dictionary
        self.num_caps = 0  # initialize the value that stores the number of OpenCV captures

        self.cap_num_dict = {1: (320, 320), 2: (128, 128), 3: (96, 96)}
        self.heatmap = Heatmap()

        self.n_heatmap = 0

    def turn_on(self, filename=''):
        """
        This method loads all of the OpenCV camera captures into self.cap_dict and, and also loads all of the
        objects used for detection and stores them in self.detect_dict. One of the main purposes of this function is
        so that the camera(s) is/are not turned on until the select_feeds.html file is opened.
        :param filename: path to the video file being uploaded if not using the live camera footage
        :return:
        """
        i = 0  # counter used for indexing
        # If a filename is specified, do no camera feeds.
        if filename != '':
            cap = cv2.VideoCapture(filename)
            if cap.isOpened():
                # Representation of dictionary entries: [OpenCV capture, calibration boolean, 0 which is a placeholder
                # for the calibration matrix, 1, which is boolean for whether the camera is to be used, and (height,
                # width) of the camera frame]
                print(type(self.cap_dict))
                self.cap_dict[i] = [cap, 0, 0, 1, (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))]
                self.matrix_dict[i] = []
                print("[INFO] loading YOLO from disk for file {}...".format(filename))
                self.detect_dict[i] = cv2.dnn.readNetFromDarknet(self.configPath,
                                                                 self.weightsPath)  # load our YOLO object detector trained on COCO dataset (80 classes)
                print("[INFO] finished loading YOLO for file {}...".format(filename))
                if i == 0:  # the following part only needs to be initialized once, but it is in this for loop because
                    #  the at least one cv2.dnn.readNetFromDarknet(...) has to have been initialized for this part to
                    # be initialized.

                    # determine only the *output* layer names that we need from YOLO
                    self.ln = self.detect_dict[i].getLayerNames()
                    self.ln = [self.ln[i[0] - 1] for i in self.detect_dict[i].getUnconnectedOutLayers()]
                    self.num_caps = 1
                    return
            else:  # The file did not load correctly
                # TODO: Handle this case better
                self.num_caps = 0
                return
        else:  # using live footage from the cameras connected to the computer
            # add all of the OpenCV captures to self.cap_dict:
            while i < 5:  # support up to 5 different cameras
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Representation of dictionary entries: [OpenCV capture, calibration boolean, 0 which is a placeholder
                    # for the calibration matrix, 1, which is boolean for whether the camera is to be used, and (height,
                    # width) of the camera frame]
                    self.cap_dict[i] = [cap, 0, 0, 1, (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))]
                    print("[INFO] loading YOLO from disk for camera {}...".format(i))
                    self.detect_dict[i] = cv2.dnn.readNetFromDarknet(self.configPath,
                                                                     self.weightsPath)  # load our YOLO object detector trained on COCO dataset (80 classes)
                    print("[INFO] finished loading YOLO for camera {}...".format(i))
                    if i == 0:  # the following part only needs to be initialized once, but it is in this for loop because
                        #  the at least one cv2.dnn.readNetFromDarknet(...) has to have been initialized for this part to
                        # be initialized.

                        # determine only the *output* layer names that we need from YOLO
                        self.ln = self.detect_dict[i].getLayerNames()
                        self.ln = [self.ln[i[0] - 1] for i in self.detect_dict[i].getUnconnectedOutLayers()]

                else:  # all the cameras that can be detected have been, so break the loop:
                    self.num_caps = i
                    break
                i += 1

    def turn_off(self, update_heatmaps=True, reset_detection=True):
        """
        Releases all of the OpenCV captures and clears the appropriate attributes from the class.
        :param update_heatmaps: a boolean for whether the heatmaps should also be reset
        :return: void
        """
        if update_heatmaps:
            self.heatmap.reset(self.num_caps, self.cap_dict)

        # TODO: currently not used...
        for i in range(self.num_caps):
            cap = self.cap_dict[i]
            cap[0].release()

        self.cap_dict = {}
        self.num_caps = 0

        if reset_detection:
            self.detect_dict = {}
            self.ln = 0

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

    def start_recording(self):
        """
        Called by record_button_receiver() in app.py, this sets the recording parameter equal to True and resets the
        heatmap data.
        :return: void
        """
        self.record = True
        self.heatmap.reset(self.num_caps, self.cap_dict)

    def stop_recording(self):
        """
        Called by record_button_receiver() in app.py, this sets the recording parameter equal to False and signals the
        heatmap class to record the end time of the recording.
        :return:
        """
        self.record = False
        self.heatmap.record_end_time()

    def calibrate(self, cap_num, frame):
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

        frame_x = self.cap_dict[cap_num][4][1]  # number of pixels wide the camera frame is
        frame_y = self.cap_dict[cap_num][4][0]  # number of pixels tall the camera frame is
        frame_x_c = int(frame_x / 2)  # center of the camera frame in the x direction

        # detect the ARUCO markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        len_markers = len(markers)

        if self.stopwatch.check_stopwatch() > 2:
            self.n = 0  # reset the counter if it has been too long; counter is used for ensuring enough frames have
                        # been captured for the average matrix to be calculated
            self.matrix_dict[cap_num] = []  # reset the matrix_list if this is the case

        # visualize the process of the calibration process: n frames calibrated / self.threshold needed
        cv2.rectangle(frame, (0, frame_y - 30), (int(self.n * frame_x / self.threshold), frame_y), (157, 161, 100), -1)

        if len_markers == 0:  # look for one marker
            cv2.putText(frame, "A full calibration sheet isn't visible.", (frame_x_c - int(frame_x / 10), frame_y - 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            return frame

        elif len_markers > 1:  # handle the edge case where multiple calibration markers are in frame
            cv2.putText(frame, "That's too many markers! Please only use one calibration sheet at a time.",
                        (frame_x_c - int(frame_x / 6), frame_y - 15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            return frame

        else:  # one calibration marker has been detected
            self.n += 1  # update the counter

            # To calculate the required perspective shift, we will need to find how exactly the calibration square
            # skewed by looking at the shape of the square aruco marker from the perspective of the camera. The
            # coordinates of this square as found from the 4 corners of the aruco marker
            square = []  # an ordered list that contains the coordinates of the square outlined by the ARUCO markers
            for i in range(4):
                square.append(tuple(markers[0][0][i]))

            # The following section of code Infers what the calibration square would look like if it was orthogonal to
            # the camera. This is set later as the destination square: square_dest. The destination square is used for
            # warping the camera perspective to a desireable one.

            # vector of bottom edge of the calibration square, placed at the origin:
            bottom_edge_orig = np.float32([square[2][0] - square[3][0], square[2][1] - square[3][1]])
            mbe = np.linalg.norm(bottom_edge_orig)  # length (magnitude) of the bottom edge (mbe)
            unit_bottom_edge = bottom_edge_orig / mbe  # bottom edge vector changed to length of 1
            # absolute angle between horizontal and bottom edge:
            theta = abs(acos(unit_bottom_edge.dot(np.float32([1, 0]))))
            # set up the rotation matrix:
            rotation_matrix = np.float32([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            # template for the destination square (to be modified by rotation matrix)
            orig_square_pre = np.float32([[0, 0], [mbe, 0], [mbe, mbe], [0, mbe]])
            # apply the rotation matrix to the original square, translated to match the position of observed calibration
            # square
            square_dest = orig_square_pre.dot(rotation_matrix) + np.float32([square[0][0], square[0][1]])

            # Visualization:
            if self.debug:
                # colors that will be used to test whether the order of the stored data is correct
                colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}

                # When in debug mode, the box and its corners will be shown and color-coded in this order:
                # 1) Top-left (Blue)  2) Top-right (Green)  3) Bottom-right (Red)  4) Bottom left (Cyan)
                for i in range(4):
                    marker = markers[0][0]
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
            self.matrix_dict[cap_num].append(matrix)  # record the transformation matrix for this frame
            frame = cv2.warpPerspective(frame, matrix, (frame_x, frame_y))  # apply the transformation matrix

            # take self.threshold frames to calibrate
            if self.n > self.threshold:
                # find the average calibration matrix between the self.threshold frames by summing up all of the
                # calibration matrices and performing elementwise division of the resulting matrix by the number of
                # matrices that were added.
                calibration_matrix = np.float32()
                for i in range(self.threshold):  # self.matrix_dict[cap_num] should be self.threshold long
                    calibration_matrix = calibration_matrix + self.matrix_dict[cap_num][i]
                self.cap_dict[cap_num][2] = calibration_matrix / self.threshold

                return frame
            else:
                self.stopwatch.log_time()  # log the time
                return frame

    def _parse_detected(self, frame, layerOutputs, cap_num):
        """
        This function takes the output of the object detection and parses the information down to bounding box
        coordinates of any people that the algorithm detects.

        This method was derived from the example at this link:
        https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
        credit to: Adrian Rosebrock

        :param frame: input frame to the object detection
        :param w: width of the frame
        :param h: height of the frame
        :param layerOutputs: output from the object detection
        :param LABELS: labels for the YOLO model
        :return: boxes, frame_copy: bounding boxes for the detected people, a copy of the frame with bounding boxes
        drawn
        """
        boxes = []
        confidences = []
        classIDs = []

        # frame width and height
        W = self.cap_dict[cap_num][4][1]
        H = self.cap_dict[cap_num][4][0]

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)

                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected probability is greater than the minimum
                # probability
                if confidence > DETECTION_THRESHOLD:
                    # scale the bounding box coordinates back relative to the size of the image, keeping in mind that
                    # YOLO actually returns the center (x, y)-coordinates of the bounding box followed by the boxes'
                    # width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, DETECTION_THRESHOLD, 0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                if self.LABELS[classIDs[i]] != 'person':
                    frame_copy = frame
                    return [], frame_copy
                else:
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # modify boxes such that it includes the top left and bottom right coordinates as opposed to the top
                    # left coordinates and the length and width of the box - this helps for adding information to the
                    # heatmap
                    boxes[i][2] = x + w
                    boxes[i][3] = y + h

                    # draw a bounding box rectangle and label on the image
                    # color = (157, 161, 100) # uncomment to use the turquoise color
                    color = (200, 200, 200)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # uncomment the following two lines to add a label and confidence level to the bounding box
                    # text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        frame_copy = frame
        return boxes, frame_copy

    def get_frame(self, cap_num, calibrate=False):
        """
        Returns the captured frame that is warped by the calibration_matrix
        :param: cap_num - index number of the OpenCV capture
        :param: calibrate - boolean for whether the frame should be perspective corrected
        :return: frame or frame converted to bytes, depending on use case
        """
        # if the camera is set to off, then dim the frame by x0.2

        cap = self.cap_dict.get(cap_num)[0]  # select the camera from self.cap_dict
        _, frame = cap.read()  # read the camera capture

        self.n_heatmap = self.heatmap.n  # reset the variable that selects the heatmap to the current one as defined by

        # pull out the height and width of the camera frame
        height = self.cap_dict[cap_num][4][0]
        width = self.cap_dict[cap_num][4][1]

        if self.cap_dict[cap_num][3] == 0:  # if the camera is muted
            frame = frame * 0.2  # dim the camera feed
            return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()  # perform no further computation

        else:  # if the camera is not muted:
            if self.cap_dict[cap_num][1] == 1 or calibrate == True:  # if the camera is in calibration mode:
                if type(self.cap_dict[cap_num][2]) != int:  # already calibrated if true; compare type because when it
                    # becomes the calibration matrix, the truth value of a multi-element array is ambiguous
                    frame = cv2.warpPerspective(frame, self.cap_dict[cap_num][2], (width, height))
                else:  # perform calibration
                    frame = self.calibrate(cap_num, frame)
                    return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

            # Perform the person detection
            net = self.detect_dict[cap_num]  # select the image processor (net)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.cap_num_dict[self.num_caps],
                                         swapRB=True, crop=False)  # pre=process the image for detection

            # run detection on the frame:
            net.setInput(blob)
            layerOutputs = net.forward(self.ln)
            boxes, frame = self._parse_detected(frame, layerOutputs, cap_num)

            # add bounding boxes to the heatmap if in recording mode
            if self.record:
                for i in range(len(boxes)):
                    self.heatmap.add_to_heatmap(cap_num, boxes[i])

                heat_overlay = self.heatmap.return_heatmap(cap_num)
                frame = cv2.addWeighted(frame, 0.4, heat_overlay, 0.6, 0)

            return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

    def show_heatmap(self, cap_num):
        """
        This is a parred-down version of get_frame() that is used by the get_results page to display a heatmap on top
        of the camera frame.
        :param cap_num: index of the camera desired
        :return: frame to be displayed
        """
        heat_overlay = self.heatmap.return_heatmap(cap_num, self.n_heatmap)  # selects the heatmap of interest

        cap = self.cap_dict.get(cap_num)[0]  # select the camera
        _, frame = cap.read()
        frame = cv2.addWeighted(frame, 0.2, heat_overlay, 0.8, 0)  # overlay the heatmap on to the frame
        return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()


class Stopwatch:
    """
    Stopwatch is utility class used by the calibrate() method of the ProcessingEngine class; acts as a stopwatch.
    """

    def __init__(self):
        self.logged_time = time.time()
        self.time_since_last_check = 0

    def log_time(self):
        """
        Record the current time
        :return: void
        """
        self.logged_time = time.time()

    def check_stopwatch(self):
        """
        Check the elapsed time between when this method is called and when the stopwatch time was last logged.
        :return: self.time_since_last_check
        """
        self.time_since_last_check = time.time() - self.logged_time
        return self.time_since_last_check


if __name__ == "__main__":
    engine = ProcessingEngine(debug=True)
    engine.turn_on()
    engine.record = True
    # display each camera connected to the computer with a corrected perspective
    while True:
        for cap_num in range(engine.num_caps):
            print('here')
            frame = engine.get_frame(cap_num, calibrate=False)
            cv2.imshow("frame {}".format(cap_num), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                engine.heatmap.show_heatmap(cap_num)
                break
