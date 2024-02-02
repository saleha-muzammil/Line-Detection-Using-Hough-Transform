# Line Detection using Hough Transform Algorithm

This repository contains a Python implementation of the Hough Transform algorithm for detecting lines in images. The primary goal is to identify straight lines present in the input image using edge detection and Hough Transform.

## Requirements

- Python 3.x
- NumPy
- ImageIO
- OpenCV
- Matplotlib
- Tkinter (for the GUI)
- Pillow (PIL)

## Installation

```bash
pip install numpy imageio opencv-python matplotlib Pillow
```
## Algorithm Overview

### Hough Transform

The Hough Transform is a technique for identifying shapes within an image, particularly used for detecting lines. In the context of line detection, the Hough Transform represents each pixel in the input image as a point in a parameter space, where each point corresponds to a potential line in the image.

The general equation for a line in polar coordinates is given by:

\[ r = x \cos(\theta) + y \sin(\theta) \]

Here, \(r\) is the perpendicular distance from the origin to the line, and \(\theta\) is the angle formed by the perpendicular line and the x-axis.

In the algorithm, the accumulator array is used to count the intersections of curves in the parameter space, and peaks in this accumulator array correspond to potential lines in the image.

The implementation involves the following steps:

- **Edge Detection:** Apply the Canny edge detection algorithm to identify potential edges in the input image.
  
- **Hough Transform:** Transform the edge-detected image using the Hough Transform, filling an accumulator array representing the parameter space for lines.
  
- **Peak Identification:** Identify peaks in the accumulator array, corresponding to potential lines in the image.
  
- **Line Segments Identification:** Extract line segments based on the identified peaks.
  
- **Line Connection and Merging:** Connect and merge line segments based on their proximity and orientation.
  
- **Result Visualization:** Generate the final image with detected lines for visualization.

## GUI Usage

To use the graphical user interface (GUI), run the provided script:

```bash
python3 line_detection.py
```
