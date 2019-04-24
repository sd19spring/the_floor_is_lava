"""
Web application app
"""

from flask import render_template, Response, request, json
from web_classes import WebApplication
from cv_classes import ProcessingEngine
import time


def feed(engine, cap_num):
    """
    Opens a camera reader, gets a processed frame, encodes it to JPEG, and returns it as a
    snippet of a multipart response body.
        Author of this function: Elias Gabriel
    """
    # In a loop, get the current frame of the camera as a byte sequence and yield it to the calling process.
    # This lets us send a HTTP response back to the client, but keeps it open to allow for continious
    # updates. In effect, this streams image data from the server to the client's computer through a
    # Motion JPEG.
    # Wrap the encoded frame in a multipart image section, to be inserted into the multipart HTTP response
    while True:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + engine.get_frame(cap_num) + b'\r\n')


def record_button_true_receiver(error=False):
    # read json + reply
    record = request.get_data()
    print(record)
    print('here')
    return 'none'


def index(error=False):
    """ Renders the index HTML page. """
    # Render the index page, showing the error message if something went wrong
    return render_template('index.html', visibility=("visible" if error else "hidden"))


def select_feeds(error=False):
    return render_template('select_feeds.html', NUM_CAPS=engine.num_caps - 1)


def more_info(error=False):
    return render_template('more_info.html', visibility=("visible" if error else "hidden"))


def eye(CAP_NUM):
    """ Returns a mixed multipart HTTP response containing streamed MJPEG data, pulled from
    the OpenCV image processor. """
    # if int(time.time()) % 3 == 0:
    #     engine.refresh()

    # Create and return a mutlipart HTTP response, with the separate parts defined by '--frame'
    try:
        return Response(feed(engine, CAP_NUM), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        return index(error=True)


# Create a processing engine. Although it generally isn't good practice to do this in the body of the document, the
# object needs to be a global instance so that only one is created no matter how many cameras are created. Also, the
# methods of this class are accessed across multiple functions.
engine = ProcessingEngine()

if __name__ == "__main__":
    # Create a new web application
    app = WebApplication("pedheatmap")

    # Define the application routes
    app.route({
        '/': index,
        '/select_feeds': select_feeds,
        '/more_info': more_info,
        '/<CAP_NUM>': eye,
        '/record_button_true_receiver': record_button_true_receiver}, post_only=['/record_button_true_receiver'])

    # Beginning listening on `localhost`, port 3000
    app.listen(port=8080)
