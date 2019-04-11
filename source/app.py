"""
Web application app
"""

from flask import render_template
from web_classes import WebApplication


def index(error=False):
    """ Renders the index HTML page. """
    # Render the index page, showing the error message if something went wrong
    return render_template('index.html', visibility=("visible" if error else "hidden"))


def feed(error=False):
    return render_template('feed.html', visibility=("visible" if error else "hidden"))


if __name__ == "__main__":
    # Create a new web application
    app = WebApplication("pedheatmap")
    # Define the application routes
    app.route({
        '/': index,
        '/feed': feed,
    })

    # Beginning listening on `localhost`, port 3000
    app.listen(port=8080, env="development")
