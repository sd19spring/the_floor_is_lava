# The Floor is Lava - a computer vision utility for tracking pedestrian foot traffic

### Setup:
Clone the repository and enter the root directory of the project:
```
git clone https://github.com/sd19spring/the_floor_is_lava.git; cd the_floor_is_lava
```

Make the setup script executable:
```
chmod +x setup.sh
```
Execute the setup script:
```
./setup.sh
```
This will install all of the dependencies for the project, including Python modules and the yolov3.weights file that is necessary for the person detection.

Our project makes heavy use of the following modules:
- Opencv
- Flask
The rest of the dependencies are listed in requirements.txt. You may want to use a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) when running the setup script.

### Usage
In the /source directory:
```
python app.py
```
A web app will open up and walk you through the rest!

### Features:
Gain a better understanding of your retail store by visualizing how people move through your store.
- Easy to use web app
- Handles multiple cameras, and can even stitch them together
- Robust tracking using a machine learning object detection algorithm

### Examples:
*tbd*

### Authors
* Gabriella Bourdon
* Michael Remley
* Nick Bourdon
* Duncan Mazza
