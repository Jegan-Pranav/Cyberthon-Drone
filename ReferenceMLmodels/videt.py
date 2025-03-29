import cv2
import torch
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11st.pt")  # Your trained model

# Input video path and output video settings
input_video_path = "emniyett.mp4"  # Change this to your input video file
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

# Class IDs based on your dataset
PEDESTRIAN_CLASSES = {0, 1}  # Pedestrian & People
VEHICLE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 10}  # All vehicle-related classes

# Process each frame
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
    crowd_status = "Crowd" if pedestrian_count > 10 else "No Crowd"
    traffic_status = "Traffic" if vehicle_count > 20 else "No Traffic"

    cv2.putText(frame, f"Pedestrians: {pedestrian_count} ({crowd_status})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count} ({traffic_status})", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
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

