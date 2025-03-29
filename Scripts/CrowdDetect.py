import cv2
import numpy as np
import zmq
import torch
import time
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis.log'),
        logging.StreamHandler()
    ]
)

# Constants
PEDESTRIAN_CLASSES = {0, 1}
VEHICLE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 10}
DISPLAY_W, DISPLAY_H = 1280, 720
MAX_FPS = 30
GPU_CLEANUP_INTERVAL = 100  # Clean GPU cache every 100 frames

# ROI setup
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
    if event == cv2.EVENT_LBUTTONDOWN:
        scaled_pts = scale_points_to_frame(roi_points_norm, DISPLAY_W, DISPLAY_H)
        for i, (px, py) in enumerate(scaled_pts):
            if abs(px - x) < 10 and abs(py - y) < 10:
                dragging_idx = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
        xn = x / float(DISPLAY_W)
        yn = y / float(DISPLAY_H)
        roi_points_norm[dragging_idx] = [xn, yn]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = None

def initialize_zmq():
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    sub_socket.set(zmq.RCVHWM, 10)  # Increased buffer size
    sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
    return context, sub_socket

def main():
    # Initialize
    model = YOLO("yolo11st.pt")
    context, sub_socket = initialize_zmq()
    
    # Create window
    cv2.namedWindow("Traffic & Crowd Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic & Crowd Analysis", DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback("Traffic & Crowd Analysis", mouse_callback)

    # Performance tracking
    frame_count = 0
    last_frame_time = time.time()
    fps_counter = 0
    fps = 0
    last_fps_time = time.time()

    try:
        while True:
            current_time = time.time()
            
            # Frame rate control
            if current_time - last_frame_time < 1./MAX_FPS:
                continue
                
            try:
                # Receive frame
                jpeg_buffer = sub_socket.recv(flags=zmq.NOBLOCK)
                frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    logging.warning("Received empty frame")
                    continue
                    
                # Update timing
                last_frame_time = current_time
                frame_count += 1
                fps_counter += 1
                
                # Calculate FPS
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    last_fps_time = current_time
                    logging.info(f"Processing FPS: {fps}")
                
                # Perform detection
                try:
                    results = model(frame, imgsz=640)[0]
                except Exception as e:
                    logging.error(f"Detection failed: {str(e)}")
                    continue
                
                # Process detections
                orig_height, orig_width = frame.shape[:2]
                roi_pts_orig = scale_points_to_frame(roi_points_norm, orig_width, orig_height)
                mask_orig = create_mask(roi_pts_orig, orig_width, orig_height)
                
                pedestrian_count = 0
                vehicle_count = 0
                
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
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add overlay
                crowd_status = "More Crowd" if pedestrian_count > 20 else "Less Crowd" if pedestrian_count > 0 else "No Crowd"
                traffic_status = "More Traffic" if vehicle_count > 20 else "Less Traffic" if vehicle_count > 0 else "No Traffic"
                
                cv2.putText(frame, f"Pedestrians: {pedestrian_count} ({crowd_status})", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicles: {vehicle_count} ({traffic_status})", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"FPS: {fps}", (50, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display
                display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                overlay = display_frame.copy()
                roi_pts_disp = scale_points_to_frame(roi_points_norm, DISPLAY_W, DISPLAY_H)
                cv2.polylines(overlay, [roi_pts_disp], True, (255, 255, 0), 2)
                for (dx, dy) in roi_pts_disp:
                    cv2.circle(overlay, (dx, dy), 7, (0, 255, 255), -1)
                
                display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
                cv2.imshow("Traffic & Crowd Analysis", display_frame)
                
                # Periodic cleanup
                if frame_count % GPU_CLEANUP_INTERVAL == 0:
                    torch.cuda.empty_cache()
                    logging.info("Performed GPU cache cleanup")
                
                # Exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty("Traffic & Crowd Analysis", cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
            except zmq.Again:
                continue
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                break
                
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    finally:
        cv2.destroyAllWindows()
        sub_socket.close()
        context.term()
        logging.info("Application shutdown complete")

if __name__ == "__main__":
    logging.info("Starting Traffic & Crowd Analysis")
    main()
