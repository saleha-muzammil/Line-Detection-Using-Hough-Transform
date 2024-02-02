import numpy as np
import imageio
import math
import cv2
import matplotlib.pyplot as plt

class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.angle = math.atan2(y2 - y1, x2 - x1)
        self.length = math.hypot(x2 - x1, y2 - y1)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def can_connect(line1, line2, connect_distance, angle_threshold=np.deg2rad(10)):
    # Check if the angle between two line segments is within the threshold
    if abs(line1.angle - line2.angle) > angle_threshold:
        return False
    # Check if any end points of line2 are close enough to the endpoints of line1
    return any(math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]) < connect_distance
               for pt1 in [(line1.x1, line1.y1), (line1.x2, line1.y2)]
               for pt2 in [(line2.x1, line2.y1), (line2.x2, line2.y2)])

def merge_lines(line1, line2):
    endpoints = [(line1.x1, line1.y1), (line1.x2, line1.y2), (line2.x1, line2.y1), (line2.x2, line2.y2)]
    endpoints.sort(key=lambda p: math.atan2(p[1] - line1.y1, p[0] - line1.x1))
    line1.x1, line1.y1 = endpoints[0]
    line1.x2, line1.y2 = endpoints[-1]
    line1.angle = math.atan2(line1.y2 - line1.y1, line1.x2 - line1.x1)
    line1.length = math.hypot(line1.x2 - line1.x1, line1.y2 - line1.y1)

def detect_lines(img, num_peaks, connect_distance, min_segment_length, value_threshold):
    edges = cv2.Canny(img, 100, 300, apertureSize=3)
    accumulator, thetas, rhos = hough_transform(edges, value_threshold)
    lines = identify_lines(edges.shape, accumulator, thetas, rhos, num_peaks)
    connected_lines = connect_and_merge_lines(lines, connect_distance, min_segment_length)
    return draw_lines_on_image(img.shape, connected_lines)

def hough_transform(img, value_threshold):
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
    diag_len = int(round(math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.uint8)
    are_edges = img > value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)
    for x, y in zip(x_idxs, y_idxs):
        for t_idx, (cos_t, sin_t) in enumerate(zip(np.cos(thetas), np.sin(thetas))):
            rho = diag_len + int(round(x * cos_t + y * sin_t))
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos

def identify_lines(shape, accumulator, thetas, rhos, num_peaks):
    lines = []
    indices = np.argpartition(accumulator.flatten(), -num_peaks)[-num_peaks:]
    peak_rhos, peak_thetas = np.unravel_index(indices, accumulator.shape)
    diag_len = int(math.sqrt(shape[0]**2 + shape[1]**2))
    for r, theta in zip(peak_rhos, peak_thetas):
        a, b = np.cos(thetas[theta]), np.sin(thetas[theta])
        x0, y0 = a * rhos[r], b * rhos[r]
        x1, y1 = int(x0 + diag_len * (-b)), int(y0 + diag_len * (a))
        x2, y2 = int(x0 - diag_len * (-b)), int(y0 - diag_len * (a))
        lines.append(LineSegment(x1, y1, x2, y2))
    return lines

def connect_and_merge_lines(lines, connect_distance, min_segment_length):
    for i, line1 in enumerate(lines):
        if line1 is None: continue
        for j, line2 in enumerate(lines[i+1:], start=i+1):
            if line2 and can_connect(line1, line2, connect_distance):
                merge_lines(line1, line2)
                lines[j] = None
    return [line for line in lines if line and line.length >= min_segment_length]

def draw_lines_on_image(shape, lines):
    img_with_lines = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        cv2.line(img_with_lines, (line.x1, line.y1), (line.x2, line.y2), 255, 1)
    return img_with_lines

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Add the image processing functions (rgb2gray, detect_lines, etc.) here...

class LineDetectionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Line Detection GUI")

        # Layout configuration
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(2, weight=1)

        # Canvas for the original image
        self.img_canvas = tk.Canvas(master, width=300, height=300, bg="grey")
        self.img_canvas.grid(row=0, column=0, sticky="nsew")

        # Canvas for the edge-detected image
        self.edge_canvas = tk.Canvas(master, width=300, height=300, bg="grey")
        self.edge_canvas.grid(row=0, column=1, sticky="nsew")

        # Canvas for the final image with detected lines
        self.result_canvas = tk.Canvas(master, width=300, height=300, bg="grey")
        self.result_canvas.grid(row=0, column=2, sticky="nsew")

        # Buttons for actions
        open_button = tk.Button(master, text="Open Image", command=self.open_image)
        open_button.grid(row=1, column=0, sticky="ew")

        detect_edges_button = tk.Button(master, text="Detect Edges", command=self.detect_edges)
        detect_edges_button.grid(row=1, column=1, sticky="ew")

        process_button = tk.Button(master, text="Detect Lines", command=self.process_image)
        process_button.grid(row=1, column=2, sticky="ew")

        # Image data
        self.original_image = None
        self.edge_image = None
        self.processed_image = None

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = Image.open(file_path)
            self.original_image.thumbnail((300, 300))
            self.display_image(self.original_image, self.img_canvas)

    def detect_edges(self):
        if self.original_image:
            img_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_cv, 150, 300)
            self.edge_image = Image.fromarray(edges)
            self.display_image(self.edge_image, self.edge_canvas)
        else:
            messagebox.showinfo("Info", "Please open an image first.")

    def process_image(self):
        if self.edge_image:
            img_cv = np.array(self.edge_image)
            # Assuming detect_lines is a function you have defined
            line_image = detect_lines(img_cv, num_peaks=90, connect_distance=50, min_segment_length=10, value_threshold=50)
            self.processed_image = Image.fromarray(line_image)
            self.display_image(self.processed_image, self.result_canvas)
        else:
            messagebox.showinfo("Info", "Please detect edges first.")

    def display_image(self, image, canvas):
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=image_tk)
        canvas.image = image_tk  # Keep a reference to prevent garbage-collection

    def run(self):
        self.master.mainloop()

if __name__ == '__main__':
    root = tk.Tk()
    app = LineDetectionGUI(root)
    app.run()

