"""
Contains classes used for the web application.

For the final project by Duncan, Michael, Nick, and Gabriella of the Software Design class at Olin College of
Engineering in the spring of 2019.

@Author: Elias Gabriel (used with permission for this project; some modifications were made)
"""


from flask import Flask
import os
from flask_session import Session


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
        Session(self)  # for the cookies

    def listen(self, **options):
        """ Asks Flask to begin listening to HTTP requests, with options if given. """
        # If a host is not given, assume localhost
        self.run(options.get('host', "127.0.0.1"), options.get('port', 3000), options)

    def route(self, routes):
        """ Registers each URL rule in routes to its specificed endpoint and response function. """
        for url, func in routes.items():
            self.add_url_rule(url, func.__name__, func, methods=['GET', 'POST'])
