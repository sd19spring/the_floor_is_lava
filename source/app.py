"""
Web application app
"""

from flask import render_template, Response, request, jsonify
from api.web_classes import WebApplication
from api.cv_classes import ProcessingEngine
import webbrowser


def record_button_receiver():
    """
    Listens for whether the button to trigger recording the footage on the web app is pressed, and instructs the engine
    class accordingly.
    :return: 'none'
    """
    record = request.get_data().decode()

    # flip the switch
    if record == 'true':
        engine.start_recording()
    else:
        engine.stop_recording()
        send_recording_info()

    # return 'none' and not None because Flask doesn't like it when None is returned.
    return 'none'


def reset_button_receiver():
    print('here ############################3')
    turn_on_bool = request.get_data().decode()
    print(turn_on_bool)
    if turn_on_bool == 'false':
        engine.reset_all(turn_on=False)
    else:
        engine.reset_all()

    # return 'none' and not None because Flask doesn't like it when None is returned.
    return 'none'


def send_recording_info():
    if engine.heatmap.n == -1:
        text = "No recordings yet. Statistics about your recordings will show up here."
    else:
        text = "Recording #{}\n ".format(engine.n_heatmap + 1) + engine.heatmap.get_time_info(engine.n_heatmap)
    return jsonify(text=text)


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


def increment_heatmap_display():
    """
    Used for switching between which heatmap is shown in the results page.
    :return: void
    """
    increment = request.get_data().decode()
    if increment == 'up':
        if not engine.n_heatmap >= engine.heatmap.n:
            engine.n_heatmap += 1
    else:
        if not engine.n_heatmap == 0:
            engine.n_heatmap -= 1

    # return 'none' and not None because Flask doesn't like it when None is returned.
    return 'none'


def index():
    return render_template('index.html', visibility=("visible" if engine.num_caps > 0 else "hidden"))


def select_feeds(error=False):
    print(engine.num_caps)
    if engine.num_caps == 0:
        engine.turn_on()
    return render_template('select_feeds.html', NUM_CAPS=engine.num_caps - 1)


def all_cam_switch():
    print('here')
    switch = False if int(request.get_data().decode()) == 1 else True  # switch is a string that represents a dictionary
    if switch:
        engine.turn_on()
    else:
        engine.turn_off()


def more_info(error=False):
    return render_template('more_info.html')


def results():
    return render_template('results.html', NUM_CAPS=engine.num_caps - 1)


def eye(CAP_NUM):
    """ Returns a mixed multipart HTTP response containing streamed MJPEG data, pulled from
    the OpenCV image processor. """

    # Create and return a mutlipart HTTP response, with the separate parts defined by '--frame'
    try:
        return Response(feed(engine, CAP_NUM), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        return index(error=True)


def feed(engine, cap_num):
    """
    Opens a camera reader, gets a processed frame, encodes it to JPEG, and returns it as a
    snippet of a multipart response body.
        Original author of this function: Elias Gabriel
    """
    # In a loop, get the current frame of the camera as a byte sequence and yield it to the calling process.
    # This lets us send a HTTP response back to the client, but keeps it open to allow for continious
    # updates. In effect, this streams image data from the server to the client's computer through a
    # Motion JPEG.
    # Wrap the encoded frame in a multipart image section, to be inserted into the multipart HTTP response

    while True:
        if engine.reset_completed:
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + engine.get_frame(int(cap_num)) + b'\r\n')

def heatmap(CAP_NUM):
    """ Returns a mixed multipart HTTP response containing streamed MJPEG data, pulled from
    the OpenCV image processor. """

    # Create and return a mutlipart HTTP response, with the separate parts defined by '--frame'
    try:
        return Response(heatmap_feed(engine, CAP_NUM), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        return index(error=True)


def heatmap_feed(engine, cap_num):
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
    print(cap_num)
    while True:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + engine.show_heatmap(int(cap_num)) + b'\r\n')


def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("./videos/" + f.filename)
        # Clear camera feeds from dict, reinit with file as source
        engine.turn_off()
        engine.turn_on("./videos/" + f.filename)
        # Update engine.num_caps
        return render_template('select_feeds.html', NUM_CAPS=engine.num_caps - 1)


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
        '/record_button_receiver': record_button_receiver,
        '/reset_button_receiver': reset_button_receiver,
        '/cap_switch': cap_switch,
        '/calib_switch': calib_switch,
        '/<CAP_NUM>': eye,
        '/<CAP_NUM>heatmap': heatmap,
        '/results': results,
        '/all_cam_switch': all_cam_switch,
        '/send_recording_info': send_recording_info,
        '/uploader': upload_file,
        '/increment_heatmap_display': increment_heatmap_display})

    webbrowser.open('http://127.0.0.1:8080/')
    # Beginning listening on `localhost`, port 8080
    app.listen(port=8080)
