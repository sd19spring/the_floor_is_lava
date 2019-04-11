"""
Contains all the classes used for computer vision and backend processing.

@author: Duncan Mazza
@revision: v1.3
"""
import cv2
import cv2.aruco as aruco
import os
import numpy as np
from math import acos, cos, sin
import imutils
import time

FACE_SCL = 4  # coefficient to scale the size of the face relative to the width of the aruco code

# for visualizing the corners of the markers
colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}


def find_square_dest(square):
    """
    Finds the smallest square that will fit over the input square
    :param square: list of 4 tuples (coordinates)
    :return: np.float32 list of 4 tuples (coordinates)

    """
    # vector of bottom edge of the calibration square, placed at the origin
    bottom_edge_orig = np.float32([square[2][0] - square[3][0], square[2][1] - square[3][1]])
    mbe = np.linalg.norm(bottom_edge_orig)  # magnitude of the bottom edge
    unit_bottom_edge = bottom_edge_orig / mbe
    theta = acos(unit_bottom_edge.dot(np.float32([1, 0])))  # angle between horiz and bottom edge
    if bottom_edge_orig[1] > 0:
        theta = -theta

    rotation_matrix = np.float32([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    orig_square_pre = np.float32([[0, 0], [mbe, 0], [mbe, mbe], [0, mbe]])
    return orig_square_pre.dot(rotation_matrix) + np.float32([square[0][0], square[0][1]])


class Stopwatch:
    def __init__(self):
        self.logged_time= time.time()
        self.time_since_last_check = 0

    def log_time(self):
        self.logged_time = time.time()

    def check_stopwatch(self):
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


class ProcessingEngine:
    """
    The main backend class for image per-processing, appending given images to tracked positional
    ARCUO markers in a media feed.
    """

    def __init__(self, source, threshold=60, debug=False):
        self.is_calibrated = False  # boolean for whether the camera has been calibrated
        self.n = 0  # counter for the calibration process
        self.stopwatch = Stopwatch()
        self.matrix_list = []
        self.calibration_matrix = np.float32()
        self.threshold = threshold

        # initiate parameters for aruco detection
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.debug = debug
        self.file_type = False
        self.parameters = aruco.DetectorParameters_create()

        # Set up OpenCV. If the source is local, open a local camera feed. If it is a remote
        # source, create an empty byte feed.
        if source == "local":
            self.cap0 = cv2.VideoCapture(0)
            self.cap1 = cv2.VideoCapture(1)
        elif source == "remote":
            self.cap = ByteCapture()  # for future capabilities
        # Throw an error if something isn't write
        else:
            raise ("Unknown source type! Must be `local`, `file`, or `remote`.")

    @staticmethod
    def generate_markers(num_markers):
        """ Generate ARUCO markers if they do not exist, relative to the path of execution. """
        # Create aruco markers if necessary
        if not os.path.exists("./source/static/markers"):
            os.system("mkdir ./source/static/markers")

            # Create 10 unique markers
            for marker_num in range(num_markers):
                img = aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_6X6_250), marker_num, 700)
                cv2.imwrite("./source/static/markers/marker_{}.jpg".format(str(marker_num)), img)

    def calibrate(self):
        """ Reads a frame from the given capture device, identifies the markers and inserts the desired
        faces. The processed frame is encoded as a JPEG and returned as a byte sequence. """
        # ensure that the face is already set
        _, frame = self.cap1.read()

        # frame = cv2.flip(frame_rl, 1)
        frame_x = frame.shape[1]  # number of pixels wide the camera frame is
        frame_y = frame.shape[0]  # number of pixels tall the camera frame is
        frame_x_c = int(frame_x / 2)  # center of the camera frame in the x direction
        # frame_y_c = int(frame_y / 2)  # center of the camera frame in the y direction

        # detect the aruco markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if self.stopwatch.check_stopwatch() > 2:
            self.n = 0  # reset the counter if it has been too long
            self.matrix_list = []  # reset the matrix_list

        # look for one marker sheet
        if len(markers) < 4:
            cv2.putText(frame, "A full calibration sheet isn't visible.", (frame_x_c - int(frame_x / 10), frame_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # apply the text

            if self.debug:  # visualizing the progress of the calibration
                cv2.rectangle(frame, (0, frame_y - 10), (int(self.n * frame_x / self.threshold), frame_y), (0, 255, 0), -1)

            return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

        elif len(markers) > 4:
            cv2.putText(frame, "Whoa that's too many markers! Please only use one calibration sheet at a time.",
                        (frame_x_c - int(frame_x / 6), frame_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)  # apply the text

            if self.debug:  # visualizing the progress of the calibration
                cv2.rectangle(frame, (0, frame_y - 10), (int(self.n * frame_x / self.threshold), frame_y), (0, 255, 0), -1)

            return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

        else:
            self.n += 1  # update the counter
            # make a dictionary of the markers so that they can be accessed by their id
            markers_dict = {}
            for i in range(4):
                markers_dict[ids[i][0]] = markers[i][0]

            # one calibration sheet is detected: len(markers) == 4
            # For visualizing the corners: When oriented upright, the corners of the markers will be listed in this
            # order: 1) Top-left  2) Top-right  3) Bottom-right  4) Bottom left
            if self.debug:
                for i in range(4):
                    marker = markers_dict[i]
                    # print(markers)
                    for j in range(4):
                        corner = marker[j]
                        cv2.circle(frame, tuple(corner), 10, colors[j], -1)
                    cv2.putText(frame, "{}".format(i), (int(marker[0][0] + 20), int(marker[0][1] + 20)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

            # To calculate the required perspective shift, we will need to find how a square lying flat on a surface is
            # skewed. The coordinates of this square as found from a grid of 4 aruco markers arranged in a square shape
            # are defined here.
            square = []  # contains the coordinates of the square outlined by the aruco markers
            for i in range(4):
                marker = markers_dict[i]
                square.append(tuple(marker[i]))

            square_dest = find_square_dest(square)

            # visualize the destination square (should appear distorted on the perspective-corrected frame)
            if self.debug:
                for i in range(4):
                    if i is not 3:
                        cv2.line(frame, square[i], square[i + 1], colors[i], 3)
                        cv2.line(frame, tuple(square_dest[i]), tuple(square_dest[i + 1]), colors[i], 3)
                    else:
                        cv2.line(frame, square[i], square[0], colors[i], 3)
                        cv2.line(frame, tuple(square_dest[i]), tuple(square_dest[0]), colors[i], 3)

                # visualize the process of the calibration process: n frames calibrated / self.threshold needed
                cv2.rectangle(frame, (0, frame_y - 10), (int(self.n * frame_x / self.threshold), frame_y), (0, 255, 0), -1)

            matrix = cv2.getPerspectiveTransform(np.float32(square), square_dest)  # get the transformation matrix
            self.matrix_list.append(matrix)
            frame = cv2.warpPerspective(frame, matrix, (frame_x, frame_y))

            if self.n > self.threshold:  # take self.threshold frames to calibrate; if calibrated, flip the is_calibrated switch
                self.is_calibrated = True

                # find the average calibration matrix between the self.threshold frames
                for i in range(self.threshold):  # self.matrix_list should be self.threshold long
                    self.calibration_matrix = self.calibration_matrix + self.matrix_list[i]
                self.calibration_matrix = self.calibration_matrix / self.threshold

                return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()
            else:
                self.stopwatch.log_time()  # log the time
                return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

    def get_frame(self):
        _, frame = self.cap1.read()
        frame = cv2.warpPerspective(frame, self.calibration_matrix, (frame.shape[1], frame.shape[0]))
        return frame if self.debug else cv2.imencode('.jpg', frame)[1].tobytes()

    def homography_test(self):
        _, frame0 = self.cap0.read()
        _, frame1 = self.cap1.read()

        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch([frame0, frame1])
        if status == 0:
            return stitched
        else:
            return frame1


if __name__ == "__main__":
    engine = ProcessingEngine(source="local", debug=True)

    while not engine.is_calibrated:
        output = engine.calibrate()
        if output is not None:
            cv2.imshow('calibrating', output)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    while True:
        cv2.imshow('frame', engine.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'): break
