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
        logging.FileHandler('border_surveillance.log'),
        logging.StreamHandler()
    ]
)

# Constants
PEDESTRIAN_CLASSES = {0, 1}
VEHICLE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9, 10}
DISPLAY_W, DISPLAY_H = 1280, 720
MAX_FPS = 30
GPU_CLEANUP_INTERVAL = 100

class BorderSurveillance:
    def __init__(self):
        self.dragging_idx = None
        self.show_roi = True
        self.roi_enabled = True
        self.roi_points_norm = np.array([
            [0.0, 1.0], [1.0, 1.0], 
            [1.0, 0.5], [0.0, 0.5]
        ], dtype=np.float32)
        
        # Initialize model and ZMQ
        self.model = YOLO("yolo11st.pt")
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5555")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.set(zmq.RCVHWM, 10)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)
        
        # Create window
        cv2.namedWindow("Border Surveillance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Border Surveillance", DISPLAY_W, DISPLAY_H)
        cv2.setMouseCallback("Border Surveillance", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        xn, yn = x / DISPLAY_W, y / DISPLAY_H
        
        if event == cv2.EVENT_LBUTTONDOWN:
            scaled_pts = self.scale_points_to_frame(self.roi_points_norm, DISPLAY_W, DISPLAY_H)
            for i, (px, py) in enumerate(scaled_pts):
                if abs(px - x) < 15 and abs(py - y) < 15:
                    self.dragging_idx = i
                    break
                    
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx is not None:
            xn = max(0.0, min(1.0, xn))
            yn = max(0.0, min(1.0, yn))
            self.roi_points_norm[self.dragging_idx] = [xn, yn]
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = None

    def scale_points_to_frame(self, pts_norm, w, h):
        return np.array([[int(x*w), int(y*h)] for x, y in pts_norm], dtype=np.int32)

    def create_mask(self, pts, w, h):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def run(self):
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
                    jpeg_buffer = self.sub_socket.recv(flags=zmq.NOBLOCK)
                    frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is None:
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
                        logging.info(f"FPS: {fps}")
                    
                    # Detection
                    results = self.model(frame, imgsz=640)[0]
                    
                    # Process results
                    h, w = frame.shape[:2]
                    roi_pts = self.scale_points_to_frame(self.roi_points_norm, w, h)
                    mask = self.create_mask(roi_pts, w, h) if self.roi_enabled else None
                    
                    pedestrians = 0
                    vehicles = 0
                    
                    for box in results.boxes.data:
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        cls = int(cls)
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                        if self.roi_enabled and mask is not None and mask[cy, cx] == 0:
                            continue
                            
                        if cls in PEDESTRIAN_CLASSES:
                            pedestrians += 1
                            color = (0, 255, 0)
                        elif cls in VEHICLE_CLASSES:
                            vehicles += 1
                            color = (0, 0, 255)
                        else:
                            continue
                            
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add overlay
                    status = [
                        "Border Surveillance System",
                        f"People: {pedestrians}",
                        f"Vehicles: {vehicles}",
                        f"FPS: {fps}",
                        "Controls: [R] ROI [S] Show/Hide [Q] Quit"
                    ]
                    
                    for i, text in enumerate(status):
                        y = 40 + i * 40
                        color = (255, 255, 255) if i != 1 else (0, 255, 0) if i != 2 else (0, 0, 255)
                        size = 0.7 if i > 0 else 0.8
                        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
                    
                    if self.show_roi and self.roi_enabled:
                        overlay = frame.copy()
                        cv2.polylines(overlay, [roi_pts], True, (0, 255, 255), 2)
                        for pt in roi_pts:
                            cv2.circle(overlay, tuple(pt), 8, (0, 165, 255), -1)
                        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                    
                    # Display
                    display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                    cv2.imshow("Border Surveillance", display)
                    
                    # Cleanup
                    if frame_count % GPU_CLEANUP_INTERVAL == 0:
                        torch.cuda.empty_cache()
                    
                    # Handle keys
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.roi_enabled = not self.roi_enabled
                    elif key == ord('s'):
                        self.show_roi = not self.show_roi
                        
                except zmq.Again:
                    continue
                except Exception as e:
                    logging.error(f"Error: {e}")
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self.sub_socket.close()
            self.context.term()
            logging.info("Shutdown complete")

if __name__ == "__main__":
    logging.info("Starting system")
    app = BorderSurveillance()
    app.run()
