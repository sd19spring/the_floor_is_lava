#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''

import sys
import cv2
import numpy as np
from skimage.measure import compare_ssim

if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        # sys.exit(1)

    # speed-up using multithreads
    cv2.setUseOptimized(True);
    # cv2.setNumThreads(4);

    # read image
    # im = cv2.imread(sys.argv[1])
    # im = cv2.imread("beach.jpg", 1)
    # resize image
    # newHeight = 600
    # # newWidth = int(im.shape[1]*200/im.shape[0])
    # newWidth = 600
    # im = cv2.resize(im, (newWidth, newHeight))
    # print(im.shape)

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


    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(frame, avg_frame, full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


        # set input image on which we will run segmentation
        ss.setBaseImage(thresh)

        # Switch to fast but low recall Selective Search method
        # if (sys.argv[3] == 'f'):
        ss.switchToSelectiveSearchFast()

        # Switch to high recall but slow Selective Search method
        # elif (sys.argv[3] == 'q'):
        # ss.switchToSelectiveSearchQuality()

        # # if argument is neither f nor q print help message
        # else"
        # print(__doc__)
        # sys.exit(1)

        # run selective search segmentation on input image
        rects = ss.process()
        print('Total Number of Region Proposals: {}'.format(len(rects)))

        # number of region proposals to show
        # numShowRects = 100
        numShowRects = 60
        # increment to increase/decrease total number
        # of reason proposals to be shown
        # increment = 50
        increment = -5

        # draw rectangle for region proposal till numShowRects
        for i, rect in enumerate(rects):
            if i < numShowRects:
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('frame2', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # while True:
        #     # create a copy of original image
        #     imOut = im.copy()
        #
        #     # itereate over all the region proposals
        #     for i, rect in enumerate(rects):
        #         # draw rectangle for region proposal till numShowRects
        #         if (i < numShowRects):
        #             x, y, w, h = rect
        #             cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        #         else:
        #             break
        #
        #     # show output
        #     cv2.imshow("Output", imOut)
        #
        #     # record key press
        #     k = cv2.waitKey(0) & 0xFF
        #
        #     # m is pressed
        #     if k == 109:
        #         # increase total number of rectangles to show by increment
        #         numShowRects += increment
        #     # l is pressed
        #     elif k == 108 and numShowRects > increment:
        #         # decrease total number of rectangles to show by increment
        #         numShowRects -= increment
        #     # q is pressed
        #     elif k == 113:
        #         break
        # # close image show window
        # cv2.destroyAllWindows()
