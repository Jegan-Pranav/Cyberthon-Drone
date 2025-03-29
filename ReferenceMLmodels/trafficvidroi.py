import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("yolo11st.pt")

# Video paths
input_video_path = "emniyett.mp4"
output_video_path = "output.avi"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 10

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

PEDESTRIAN_CLASSES = {0, 1}
VEHICLE_CLASSES    = {2, 3, 4, 5, 6, 7, 8, 9, 10}

# Desired display size (fixed)
DISPLAY_W, DISPLAY_H = 1280, 720

# Normalized ROI points (start with a simple rectangle)
roi_points_norm = np.array([
    [0.2, 1.0],
    [0.8, 1.0],
    [0.8, 0.5],
    [0.2, 0.5]
], dtype=np.float32)

dragging_idx = None

def scale_points_to_frame(roi_pts_norm, w, h):
    return np.array([[int(xn*w), int(yn*h)] for (xn, yn) in roi_pts_norm], dtype=np.int32)

def create_mask(roi_pts, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts], 255)
    return mask

def mouse_callback(event, x, y, flags, param):
    global dragging_idx, roi_points_norm
    # x,y are in the scaled display coords (1280x720)
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check near ROI points
        scaled_pts = scale_points_to_frame(roi_points_norm, DISPLAY_W, DISPLAY_H)
        for i, (px, py) in enumerate(scaled_pts):
            if abs(px - x) < 10 and abs(py - y) < 10:
                dragging_idx = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
        # Convert display coords -> normalized
        xn = x / float(DISPLAY_W)
        yn = y / float(DISPLAY_H)
        roi_points_norm[dragging_idx] = [xn, yn]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = None

cv2.namedWindow("Traffic & Crowd Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic & Crowd Analysis", DISPLAY_W, DISPLAY_H)
cv2.setMouseCallback("Traffic & Crowd Analysis", mouse_callback)

while True:
    ret, original_frame = cap.read()
    if not ret:
        break

    # Inference on the original frame
    results = model(original_frame, imgsz=640)[0]

    # Build ROI mask in original resolution
    roi_pts_orig = scale_points_to_frame(roi_points_norm, orig_width, orig_height)
    mask_orig    = create_mask(roi_pts_orig, orig_width, orig_height)

    # Count
    pedestrian_count = 0
    vehicle_count    = 0

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls = int(cls)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

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
        cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Put text
    crowd_status   = "More Crowd" if pedestrian_count > 20 else "Less Crowd" if pedestrian_count > 0 else "No Crowd"
    traffic_status = "More Traffic" if vehicle_count > 20 else "Less Traffic" if vehicle_count > 0 else "No Traffic"

    cv2.putText(original_frame, f"Pedestrians: {pedestrian_count} ({crowd_status})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(original_frame, f"Vehicles: {vehicle_count} ({traffic_status})", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize for display
    display_frame = cv2.resize(original_frame, (DISPLAY_W, DISPLAY_H))

    # Draw ROI in display coords
    overlay = display_frame.copy()
    roi_pts_disp = scale_points_to_frame(roi_points_norm, DISPLAY_W, DISPLAY_H)
    cv2.polylines(overlay, [roi_pts_disp], True, (255, 255, 0), 2)
    for (dx, dy) in roi_pts_disp:
        cv2.circle(overlay, (dx, dy), 7, (0, 255, 255), -1)

    display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)

    # Show
    cv2.imshow("Traffic & Crowd Analysis", display_frame)

    # Write original resolution to output
    out.write(original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_video_path}")

print(f"Device: {model.device}")


