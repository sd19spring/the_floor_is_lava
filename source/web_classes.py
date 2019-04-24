"""
Contains classes used for the web application.

For the final project by Duncan, Michael, Nick, and Gabriella of the Software Design class at Olin College of
Engineering in the spring of 2019.

@Author: originally Elias Gabriel (used with permission for this project; modifications were made by Duncan Mazza)
"""

from flask import Flask
import os
from flask_session import Session
from redis import Redis, ConnectionError
from cv_classes import ProcessingEngine


def launch_redis():
    """
    Launches the redis server
    :return: rs - Redis object
    """
    rs = Redis('localhost')
    try:
        rs.ping()
        return rs
    except ConnectionError:
        return False


class WebApplication(Flask):
    """
    A wrapper for a Flask application to simplify app configuration and launching.
    """

    def __init__(self, app_name=None, debug=False):
        # Call __init__ from the Flask superclass
        super().__init__(app_name or __name__)

        # Set configuration variables
        self.debug = debug
        self.config['SECRET_KEY'] = os.urandom(16)
        self.config['SESSION_REDIS'] = launch_redis()
        Session(self)  # for the cookies

    def listen(self, **options):
        """ Asks Flask to begin listening to HTTP requests, with options if given. """
        # If a host is not given, assume localhost. use_reloader=False because if not then everything breaks because
        # the cameras are accessed by OpenCV multiple times which causes the program to get stuck at initializing
        # the ProcessingEngine
        self.run(options.get('host', "127.0.0.1"), options.get('port', 3000), options, use_reloader=False)

    def route(self, routes):
        """ Registers each URL rule in routes to its specificed endpoint and response function. """
        for url, func in routes.items():
            self.add_url_rule(url, func.__name__, func, methods=['GET', 'POST'])
