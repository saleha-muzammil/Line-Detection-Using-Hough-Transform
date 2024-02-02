Line Detection using Hough Transform Algorithm

This repository contains a Python implementation of the Hough Transform algorithm for detecting lines in images. The primary goal is to identify straight lines present in the input image using edge detection and Hough Transform.
Requirements

    Python 3.x
    NumPy
    ImageIO
    OpenCV
    Matplotlib
    Tkinter (for the GUI)
    Pillow (PIL)

Installation

bash

pip install numpy imageio opencv-python matplotlib Pillow

Usage

The main functionality is encapsulated in the detect_lines method within the LineDetection class. The provided LineDetectionGUI class demonstrates the usage with a simple graphical user interface (GUI).

python

# Example Usage
import cv2
import numpy as np
from line_detection import LineDetection, LineDetectionGUI

# Load an image
image = cv2.imread('path/to/your/image.jpg')

# Create a LineDetection instance
line_detector = LineDetection()

# Detect lines in the image
result_image = line_detector.detect_lines(
    image,
    num_peaks=90,
    connect_distance=50,
    min_segment_length=10,
    value_threshold=50
)

# Display the result
cv2.imshow('Detected Lines', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Algorithm Overview
Hough Transform

The Hough Transform is a technique for identifying shapes within an image, particularly used for detecting lines. In the context of line detection, the Hough Transform represents each pixel in the input image as a point in a parameter space, where each point corresponds to a potential line in the image.

The general equation for a line in polar coordinates is given by:

r=xcos⁡(θ)+ysin⁡(θ)r=xcos(θ)+ysin(θ)

Here, rr is the perpendicular distance from the origin to the line, and θθ is the angle formed by the perpendicular line and the x-axis.

In the algorithm, the accumulator array is used to count the intersections of curves in the parameter space, and peaks in this accumulator array correspond to potential lines in the image.

The implementation involves the following steps:

    Edge Detection: Apply the Canny edge detection algorithm to identify potential edges in the input image.

    Hough Transform: Transform the edge-detected image using the Hough Transform, filling an accumulator array representing the parameter space for lines.

    Peak Identification: Identify peaks in the accumulator array, corresponding to potential lines in the image.

    Line Segments Identification: Extract line segments based on the identified peaks.

    Line Connection and Merging: Connect and merge line segments based on their proximity and orientation.

    Result Visualization: Generate the final image with detected lines for visualization.

Line Detection using Hough Transform Algorithm

This repository contains a Python implementation of the Hough Transform algorithm for detecting lines in images. The primary goal is to identify straight lines present in the input image using edge detection and Hough Transform.
Requirements

    Python 3.x
    NumPy
    ImageIO
    OpenCV
    Matplotlib
    Tkinter (for the GUI)
    Pillow (PIL)

Installation

bash

pip install numpy imageio opencv-python matplotlib Pillow

Usage

The main functionality is encapsulated in the detect_lines method within the LineDetection class. The provided LineDetectionGUI class demonstrates the usage with a simple graphical user interface (GUI).

python

# Example Usage
import cv2
import numpy as np
from line_detection import LineDetection, LineDetectionGUI

# Load an image
image = cv2.imread('path/to/your/image.jpg')

# Create a LineDetection instance
line_detector = LineDetection()

# Detect lines in the image
result_image = line_detector.detect_lines(
    image,
    num_peaks=90,
    connect_distance=50,
    min_segment_length=10,
    value_threshold=50
)

# Display the result
cv2.imshow('Detected Lines', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Algorithm Overview
Hough Transform

The Hough Transform is a technique for identifying shapes within an image, particularly used for detecting lines. In the context of line detection, the Hough Transform represents each pixel in the input image as a point in a parameter space, where each point corresponds to a potential line in the image.

The general equation for a line in polar coordinates is given by:

r=xcos⁡(θ)+ysin⁡(θ)r=xcos(θ)+ysin(θ)

Here, rr is the perpendicular distance from the origin to the line, and θθ is the angle formed by the perpendicular line and the x-axis.

In the algorithm, the accumulator array is used to count the intersections of curves in the parameter space, and peaks in this accumulator array correspond to potential lines in the image.

The implementation involves the following steps:

    Edge Detection: Apply the Canny edge detection algorithm to identify potential edges in the input image.

    Hough Transform: Transform the edge-detected image using the Hough Transform, filling an accumulator array representing the parameter space for lines.

    Peak Identification: Identify peaks in the accumulator array, corresponding to potential lines in the image.

    Line Segments Identification: Extract line segments based on the identified peaks.

    Line Connection and Merging: Connect and merge line segments based on their proximity and orientation.

    Result Visualization: Generate the final image with detected lines for visualization.

GUI Usage

To use the graphical user interface (GUI), run the provided script:

bash

python3 line_detection.py

The GUI allows you to open an image, detect edges, and then detect lines in the processed image.

    This implementation is based on the Hough Transform algorithm for line detection.
    The GUI utilizes Tkinter for the user interface.

