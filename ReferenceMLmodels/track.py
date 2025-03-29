import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolo11st.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, embedder="mobilenet", embedder_gpu=True)

# Open webcam (0 for default camera)
cap = cv2.VideoCapture("emniyett.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 object detection
    results = model(frame)[0]

    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        bbox_xywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]  # Convert to (x, y, w, h)
        detections.append((bbox_xywh, conf, int(cls)))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

