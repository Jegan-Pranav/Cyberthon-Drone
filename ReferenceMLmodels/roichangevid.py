import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11st.pt")  # Your trained model

# Input video path and output video settings
input_video_path = "emniyett.mp4"
output_video_path = "output.avi"
fps = 10  # Adjust FPS as needed

# Open video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Class IDs
PEDESTRIAN_CLASSES = {0, 1}  # Pedestrian & People
VEHICLE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 10}  # All vehicle-related classes

# Define default ROI points (quad shape)
roi_pts = np.array([
    (frame_width // 6, frame_height),  # Bottom-left
    (frame_width * 5 // 6, frame_height),  # Bottom-right
    (frame_width // 2 + 100, frame_height // 3),  # Top-right
    (frame_width // 2 - 100, frame_height // 3)  # Top-left
], dtype=np.int32)

# Variables for dragging ROI points
dragging = False
selected_point = None

def update_roi_mask():
    """Create an updated ROI mask based on the latest ROI points."""
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts], 255)
    return mask

roi_mask = update_roi_mask()

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for dragging ROI points."""
    global dragging, selected_point, roi_pts, roi_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(roi_pts):
            if abs(px - x) < 10 and abs(p	y - y) < 10:  # Small selection area
                dragging = True
                selected_point = i
                break

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        roi_pts[selected_point] = (x, y)
        roi_mask = update_roi_mask()

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_point = None

# Set up OpenCV window for full size display
cv2.namedWindow("Traffic & Crowd Analysis", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Traffic & Crowd Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Traffic & Crowd Analysis", mouse_callback)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, imgsz=640)[0]

    # Initialize counters
    pedestrian_count = 0
    vehicle_count = 0

    # Process detections
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)

        # Compute bounding box center
        bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Check if center of the detection is inside the ROI
        if roi_mask[bbox_center[1], bbox_center[0]] == 0:
            continue  # Ignore detections outside ROI

        # Count pedestrians and vehicles
        if cls in PEDESTRIAN_CLASSES:
            pedestrian_count += 1
            color = (0, 255, 0)  # Green for pedestrians
        elif cls in VEHICLE_CLASSES:
            vehicle_count += 1
            color = (0, 0, 255)  # Red for vehicles
        else:
            continue

        # Draw bounding boxes
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Display pedestrian and vehicle count on the video
    crowd_status = "More Crowd" if pedestrian_count > 20 else "Less Crowd" if pedestrian_count > 0 else "No Crowd"
    traffic_status = "More Traffic" if vehicle_count > 20 else "Less Traffic" if vehicle_count > 0 else "No Traffic"


    cv2.putText(frame, f"Pedestrians: {pedestrian_count} ({crowd_status})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count} ({traffic_status})", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw ROI overlay
    overlay = frame.copy()
    cv2.polylines(overlay, [roi_pts], isClosed=True, color=(255, 255, 0), thickness=2)
    for (x, y) in roi_pts:
        cv2.circle(overlay, (x, y), 7, (0, 255, 255), -1)  # Mark drag points

    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)  # Blend with transparency

    # Show frame in fullscreen mode
    cv2.imshow("Traffic & Crowd Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write frame to output video
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_video_path}")

