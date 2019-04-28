"""
Web application app
"""

from flask import render_template, Response, request
from web_classes import WebApplication
from cv_classes import ProcessingEngine


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
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + engine.get_frame(int(cap_num)) + b'\r\n')


def record_button_receiver():
    """
    Listens for whether the button to trigger recording the footage on the web app is pressed, and instructs the engine
    class accordingly.
    :return: 'none'
    """
    record = request.get_data().decode()
    if record == 'true':
        engine.record = True
    else:
        engine.record = False

    # return 'none' and not None because Flask doesn't like it when None is returned.
    return 'none'


def cap_switch():
    """
    Listens for whether the button to trigger recording the footage on the web app is pressed, and instructs the engine
    class accordingly.
    :return: 'none'
    """
    switch = request.get_data().decode()  # switch is a string that represents a dictionary
    [capNum_str, toggle_str] = switch.split('&')
    [_, capNum] = capNum_str.split('=')
    [_, toggle] = toggle_str.split('=')
    capNum = int(capNum)
    toggle = int(toggle)

    engine.cap_toggle(capNum, toggle)
    # return 'none' and not None because Flask doesn't like it when None is returned.
    return 'none'


def calib_switch():
    """
    Listens for whether the button to trigger the camera calibration process on the web app is pressed, and instructs
    the engine class to act accordingly.
    :return: 'none'
    """
    switch = request.get_data().decode()  # switch is a string that represents a dictionary
    [capNum_str, toggle_str] = switch.split('&')
    [_, capNum] = capNum_str.split('=')
    [_, toggle] = toggle_str.split('=')
    capNum = int(capNum)
    toggle = 0 if int(toggle) == 1 else 1  # switch the integer value used; for simplicity in the javascript, 1 was used
    # to indicate no calibration, but the engine class understands the opposite

    engine.calib_toggle(capNum, toggle)
    # return 'none' and not None because Flask doesn't like it when None is returned.
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
# methods and attributes of this class are accessed across multiple functions in this file.
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
        '/record_button_receiver': record_button_receiver,
        '/cap_switch': cap_switch,
        '/calib_switch': calib_switch})

    # Beginning listening on `localhost`, port 8080
    app.listen(port=8080)
