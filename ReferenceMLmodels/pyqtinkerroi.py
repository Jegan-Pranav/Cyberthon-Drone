"""
pyqt_yolo.py

A PyQt5-based application for:
- YOLOv8 object detection on a video
- Displaying frames in a resizable window
- Dragging ROI points that scale with the window
- Saving an annotated output video
"""

import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PyQt5 import QtCore, QtGui, QtWidgets

# -------------------------
# Configuration
# -------------------------
INPUT_VIDEO_PATH  = "emniyett.mp4"
OUTPUT_VIDEO_PATH = "output.avi"
MODEL_PATH        = "yolo11st.pt"

PEDESTRIAN_CLASSES = {0, 1}  # e.g., person, pedestrian
VEHICLE_CLASSES    = {2, 3, 4, 5, 6, 7, 8, 9, 10}  # e.g., cars, trucks, etc.

# ROI points in normalized coordinates: x=[0..1], y=[0..1]
# (These default to a simple trapezoid covering most of the frame.)
roi_points_norm = np.array([
    [1/6, 1.0],      # bottom-left
    [5/6, 1.0],      # bottom-right
    [0.5 + 0.1, 0.3],# top-right
    [0.5 - 0.1, 0.3] # top-left
], dtype=np.float32)


# -------------------------
# Helper Functions
# -------------------------
def scale_points_to_frame(roi_points, frame_w, frame_h):
    """
    Convert normalized ROI points (0..1) -> pixel coords.
    """
    scaled = []
    for (xn, yn) in roi_points:
        x_scaled = int(xn * frame_w)
        y_scaled = int(yn * frame_h)
        scaled.append([x_scaled, y_scaled])
    return np.array(scaled, dtype=np.int32)

def create_roi_mask(roi_pts_scaled, w, h):
    """
    Create a mask (h x w) with the ROI polygon filled white (255).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts_scaled], 255)
    return mask


# -------------------------
# VideoDisplay Widget
# -------------------------
class VideoDisplay(QtWidgets.QLabel):
    """
    A custom QLabel to display frames, handle mouse events for ROI dragging,
    and maintain the correct scaling.
    """

    # Signal to notify parent we want to close or quit
    closeSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)  # We'll handle scaling manually

        self.dragging_idx = None
        self.frame = None           # Last annotated frame (for display)
        self.orig_frame = None      # Original frame (for writing to video)
        self.orig_width = 0
        self.orig_height = 0

        # Normalized ROI
        self.roi_points_norm = roi_points_norm.copy()

    def set_original_size(self, w, h):
        """
        Store the original video resolution.
        """
        self.orig_width = w
        self.orig_height = h

    def update_frame(self, annotated_frame, orig_frame):
        """
        annotated_frame: a BGR image (numpy array) to display
        orig_frame: the original resolution frame (for writing out)
        """
        self.frame = annotated_frame
        self.orig_frame = orig_frame
        self.update()  # trigger paintEvent

    def paintEvent(self, event):
        """
        Called automatically when the widget needs to be redrawn.
        We'll convert self.frame to QImage and draw it.
        Then we'll draw ROI points on top.
        """
        painter = QtGui.QPainter(self)
        if self.frame is None:
            painter.fillRect(self.rect(), QtCore.Qt.black)
            return

        # Resize the BGR frame to this widget's size
        disp_w = self.width()
        disp_h = self.height()

        # Convert self.frame from BGR to RGB
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        display_resized = cv2.resize(rgb_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # Convert to QImage
        h, w, ch = display_resized.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(display_resized.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Draw the QImage
        painter.drawImage(0, 0, QtGui.QPixmap.fromImage(qimg))

        # Draw ROI points
        # We'll do a simple highlight so you see them
        roi_pts_disp = scale_points_to_frame(self.roi_points_norm, disp_w, disp_h)

        # Draw lines
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
        painter.setPen(pen)
        for i in range(len(roi_pts_disp)):
            p1 = roi_pts_disp[i]
            p2 = roi_pts_disp[(i+1) % len(roi_pts_disp)]
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        # Draw corner circles
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        painter.setBrush(brush)
        for (dx, dy) in roi_pts_disp:
            painter.drawEllipse(QtCore.QPoint(dx, dy), 7, 7)

    def mousePressEvent(self, event):
        """
        Check if we clicked near a ROI point, start dragging if so.
        """
        if event.button() == QtCore.Qt.LeftButton:
            disp_w = self.width()
            disp_h = self.height()
            roi_pts_disp = scale_points_to_frame(self.roi_points_norm, disp_w, disp_h)

            x_click = event.x()
            y_click = event.y()

            for i, (px, py) in enumerate(roi_pts_disp):
                if abs(px - x_click) < 10 and abs(py - y_click) < 10:
                    self.dragging_idx = i
                    break

    def mouseMoveEvent(self, event):
        """
        If we're dragging a ROI point, update its normalized coords.
        """
        if self.dragging_idx is not None:
            disp_w = self.width()
            disp_h = self.height()

            xn = event.x() / float(disp_w)
            yn = event.y() / float(disp_h)
            # Clamp to [0..1] to avoid going outside
            xn = max(0.0, min(1.0, xn))
            yn = max(0.0, min(1.0, yn))

            self.roi_points_norm[self.dragging_idx] = [xn, yn]
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging_idx = None

    def keyPressEvent(self, event):
        """
        If user presses 'q', we emit closeSignal.
        """
        if event.key() == QtCore.Qt.Key_Q:
            self.closeSignal.emit()

    def closeEvent(self, event):
        """
        Called when widget is closed (e.g. user hits the 'X').
        """
        self.closeSignal.emit()
        super().closeEvent(event)


# -------------------------
# Main Application Window
# -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Load YOLO model
        self.model = YOLO(MODEL_PATH)

        # Open video
        self.cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open video.")

        self.orig_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps         = self.cap.get(cv2.CAP_PROP_FPS) or 10.0

        # Setup video writer in original resolution
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, self.fps, (self.orig_width, self.orig_height))

        # Create central VideoDisplay
        self.video_widget = VideoDisplay()
        self.video_widget.set_original_size(self.orig_width, self.orig_height)
        self.setCentralWidget(self.video_widget)

        # Connect the closeSignal
        self.video_widget.closeSignal.connect(self.exit_app)

        # Setup a QTimer to read frames
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / self.fps))  # ~ match the video fps

        self.setWindowTitle("Traffic & Crowd Analysis (PyQt)")

        # Show in a decent size
        self.resize(1280, 720)

    def process_frame(self):
        """
        Reads a frame from the video, runs YOLO detection, draws bounding boxes,
        writes to output, and updates the display widget.
        """
        ret, frame = self.cap.read()
        if not ret:
            # End of video
            self.timer.stop()
            return

        # YOLO detection on original frame
        results = self.model(frame, imgsz=640)[0]

        # Build ROI mask in original resolution
        roi_pts_orig = scale_points_to_frame(self.video_widget.roi_points_norm,
                                             self.orig_width, self.orig_height)
        mask_orig = create_roi_mask(roi_pts_orig, self.orig_width, self.orig_height)

        # Count
        pedestrian_count = 0
        vehicle_count    = 0

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Check if center is inside ROI
            if mask_orig[cy, cx] == 0:
                continue

            if cls in PEDESTRIAN_CLASSES:
                pedestrian_count += 1
                color = (0, 255, 0)
            elif cls in VEHICLE_CLASSES:
                vehicle_count += 1
                color = (0, 0, 255)
            else:
                continue

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Crowd/Traffic statuses
        crowd_status = "More Crowd" if pedestrian_count > 20 else \
                       "Less Crowd" if pedestrian_count > 0 else "No Crowd"
        traffic_status = "More Traffic" if vehicle_count > 20 else \
                         "Less Traffic" if vehicle_count > 0 else "No Traffic"

        cv2.putText(frame, f"Pedestrians: {pedestrian_count} ({crowd_status})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count} ({traffic_status})",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write annotated frame to output video
        self.out.write(frame)

        # Update display widget with the same annotated frame
        self.video_widget.update_frame(frame, frame)

    def exit_app(self):
        """
        Stop the timer, release resources, close the app.
        """
        self.timer.stop()
        self.cap.release()
        self.out.release()
        self.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

