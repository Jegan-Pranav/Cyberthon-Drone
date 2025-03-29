import cv2
import numpy as np
import zmq
from ultralytics import YOLO
import datetime
import csv
import requests
import time
import threading
import os

# Configuration
FLASK_SERVER_URL = 'http://127.0.0.1:5000/update_data'
TELEMETRY_FETCH_INTERVAL = 0.1
OUTPUT_DIR = os.path.join('videos', 'HumanDetection')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp for filenames
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"detection_{current_time}.mp4"
metadata_filename = f"metadata_{current_time}.csv"

# Shared telemetry data
latest_telemetry = {
    'lat': 'NA',
    'lon': 'NA',
    'alt': 'NA',
    'bat': 'NA',
    'yaw': 'NA',
    'pitch': 'NA',
    'roll': 'NA'
}
telemetry_lock = threading.Lock()

def fetch_telemetry_from_flask():
    try:
        response = requests.get(FLASK_SERVER_URL, timeout=1)
        return response.json() if response.status_code == 200 else {}
    except requests.RequestException:
        return {}

def telemetry_thread():
    while True:
        data = fetch_telemetry_from_flask()
        with telemetry_lock:
            for key in latest_telemetry:
                if key in data:
                    latest_telemetry[key] = data[key]
        time.sleep(TELEMETRY_FETCH_INTERVAL)

def main(zmq_address="tcp://localhost:5555"):
    # Start telemetry thread
    threading.Thread(target=telemetry_thread, daemon=True).start()

    # Initialize YOLO model
    model = YOLO("yolo11n.pt")
    
    # ZMQ setup
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(zmq_address)
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    sub_socket.set(zmq.RCVHWM, 2)
    
    # Initialize video writer
    video_writer = None
    frame_size = None
    
    # Create output paths
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    metadata_path = os.path.join(OUTPUT_DIR, metadata_filename)
    
    # Metadata setup
    metadata_file = open(metadata_path, 'w', newline='')
    csv_writer = csv.writer(metadata_file)
    csv_writer.writerow(['Frame', 'Timestamp', 'Latitude', 'Longitude', 
                        'Altitude', 'Battery', 'Yaw', 'Pitch', 'Roll', 'HumanCount'])
    
    print(f"Processing connected to ZMQ at {zmq_address}")
    print(f"Output will be saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Video file: {video_filename}")
    print(f"Metadata file: {metadata_filename}")
    cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    try:
        while True:
            try:
                # Receive frame from ZMQ
                jpeg_buffer = sub_socket.recv(flags=zmq.NOBLOCK)
                frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Initialize video writer after first frame
                if video_writer is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, frame_size)
                    print(f"Video writer initialized")
                
                # Get telemetry data
                with telemetry_lock:
                    tel = latest_telemetry.copy()
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # Run detection
                results = model(frame, imgsz=640)[0]
                people_count = 0
                
                # Draw bounding boxes
                for box in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    cls = int(cls)
                    if cls in {0, 1}:  # Person classes
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        people_count += 1
                
                # Simplified overlay - only timestamp and GPS
                cv2.putText(frame, f"{timestamp[:-3]}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"{tel['lat']}, {tel['lon']}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write to video file
                video_writer.write(frame)
                
                # Write complete metadata
                csv_writer.writerow([
                    frame_count, timestamp, 
                    tel['lat'], tel['lon'], tel['alt'],
                    tel['bat'], tel['yaw'], tel['pitch'], tel['roll'],
                    people_count
                ])
                frame_count += 1
                
                # Show frame
                cv2.imshow("Processed Video", frame)
                
                # Exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty("Processed Video", cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
            except zmq.Again:
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
                
    finally:
        if video_writer is not None:
            video_writer.release()
        metadata_file.close()
        sub_socket.close()
        context.term()
        cv2.destroyAllWindows()
        print(f"\nProcessing complete. Files saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"- Video: {os.path.abspath(video_path)}")
        print(f"- Metadata: {os.path.abspath(metadata_path)}")

if __name__ == "__main__":
    import sys
    zmq_address = sys.argv[1] if len(sys.argv) > 1 else "tcp://localhost:5555"
    main(zmq_address)
