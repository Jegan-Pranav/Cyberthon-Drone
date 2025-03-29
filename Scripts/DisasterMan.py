import cv2
import numpy as np
import zmq
from ultralytics import YOLO
import time
import datetime
import os

class VisDroneDetector:
    def __init__(self):
        # Load YOLOv11 model
        self.model = YOLO("yolo11st.pt")
        
        # VisDrone class names and colors
        self.visdrone_classes = {
            0: 'pedestrian',
            1: 'people',
            2: 'bicycle',
            3: 'car',
            4: 'van',
            5: 'truck',
            6: 'tricycle',
            7: 'awning-tricycle',
            8: 'bus',
            9: 'motor'
        }
        
        self.class_colors = {
            0: (255, 0, 0),    # red
            1: (0, 255, 0),    # green
            2: (0, 0, 255),    # blue
            3: (255, 255, 0),  # yellow
            4: (255, 0, 255),  # magenta
            5: (0, 255, 255),  # cyan
            6: (128, 0, 128),  # purple
            7: (0, 128, 128),  # teal
            8: (128, 128, 0),  # olive
            9: (128, 0, 0)     # maroon
        }
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5555")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.set(zmq.RCVHWM, 10)  # Buffer size
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        
        # Object counters
        self.class_counts = {class_id: 0 for class_id in self.visdrone_classes}
        
        # Video writer setup
        self.video_writer = None
        self.frame_size = None
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join('..', 'videos', f'VisDrone')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create window
        cv2.namedWindow("VisDrone Detection", cv2.WINDOW_NORMAL)
        
        print("Waiting for video stream from Flask server...")
        print(f"Output will be saved to: {os.path.abspath(self.output_dir)}")

    def initialize_video_writer(self, frame):
        """Initialize video writer with the first frame's dimensions"""
        self.frame_size = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(self.output_dir, f'{self.timestamp}.mp4')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, self.frame_size)
        print(f"Video recording started: {video_path}")

    def process_frame(self, frame):
        # Reset counters
        for class_id in self.class_counts:
            self.class_counts[class_id] = 0
        
        # Run inference
        results = self.model(frame, imgsz=640)[0]
        
        # Process detections
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)
            if cls in self.visdrone_classes:
                self.class_counts[cls] += 1
                color = self.class_colors[cls]
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Label with class and confidence
                label = f"{self.visdrone_classes[cls]} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display counts
        y_offset = 30
        for class_id, count in self.class_counts.items():
            if count > 0:
                class_name = self.visdrone_classes[class_id]
                color = self.class_colors[class_id]
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
        
        return frame

    def run(self):
        try:
            while True:
                try:
                    # Get frame from ZMQ
                    jpeg_buffer = self.sub_socket.recv(flags=zmq.NOBLOCK)
                    frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                    
                    # Initialize video writer on first frame
                    if self.video_writer is None:
                        self.initialize_video_writer(frame)
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Write to video file
                    self.video_writer.write(processed_frame)
                    
                    # Display
                    cv2.imshow("VisDrone Detection", processed_frame)
                    
                    # Exit on 'q' or window close
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if cv2.getWindowProperty("VisDrone Detection", cv2.WND_PROP_VISIBLE) < 1:
                        break
                        
                except zmq.Again:
                    continue  # No frame available yet
                except Exception as e:
                    print(f"Error: {e}")
                    break
                    
        finally:
            # Cleanup
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            self.sub_socket.close()
            self.context.term()
            print(f"Video saved to: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    detector = VisDroneDetector()
    detector.run()
