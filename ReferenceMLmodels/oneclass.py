import cv2
import torch
import random
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11st.pt")  # Your trained model

# Input video path
input_video_path = "emniyett.mp4"  # Change this to your video file
output_video_path = "output.avi"

# Open video capture
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Use 30 if FPS is unavailable

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

# Define target classes (CHANGE THIS)
target_classes = {0, 1}  # Example: Only detect classes 0, 2, and 5

# Assign a unique color to each class
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class_colors = {cls: generate_random_color() for cls in target_classes}

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, imgsz=640)[0]  # Optimize inference size

    # Draw bounding boxes for target classes
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)
        if cls in target_classes:  # Only process specified classes
            color = class_colors[cls]  # Get color for class
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"Class {cls} ", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame in real-time
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_video_path}")

