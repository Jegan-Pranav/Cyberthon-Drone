import cv2
import torch
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11st.pt")  # Your trained model

# Input image directory and output video settings
image_dir = "/home/jegan/Desktop/Drone/datasets/VisDrone/VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v"
output_video_path = "output.avi"
fps = 10  # Adjust FPS as needed

# Get sorted list of image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

# Get frame dimensions from the first image
first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width, _ = first_image.shape

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Class IDs based on your dataset
PEDESTRIAN_CLASSES = {0, 1}  # Pedestrian & People
VEHICLE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 10}  # All vehicle-related classes

# Process each image
for img_file in image_files:
    image_path = os.path.join(image_dir, img_file)
    image = cv2.imread(image_path)
    
    # Run YOLOv8 detection
    results = model(image, imgsz=640)[0]
    
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
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Display pedestrian and vehicle count on the video
    crowd_status = "Crowd" if pedestrian_count > 10 else "No Crowd"
    traffic_status = "Traffic" if vehicle_count > 20 else "No Traffic"

    cv2.putText(image, f"Pedestrians: {pedestrian_count} ({crowd_status})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Vehicles: {vehicle_count} ({traffic_status})", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Traffic & Crowd Analysis", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write frame to output video
    out.write(image)

# Cleanup
out.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_video_path}")

