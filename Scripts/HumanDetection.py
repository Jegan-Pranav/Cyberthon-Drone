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
import pywhatkit
from queue import Queue
from sort import Sort

# Configuration
FLASK_SERVER_URL = 'http://127.0.0.1:5000/update_data'
TELEMETRY_FETCH_INTERVAL = 0.1
WHATSAPP_NUMBER = "+919351906003"  # Replace with recipient number
NOTIFICATION_COOLDOWN = 60  # 60 seconds between notifications

# Generate timestamp for filenames
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join('videos', f'HumanDetection')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared data
latest_telemetry = {
    'lat': 'NA', 'lon': 'NA', 'alt': 'NA',
    'bat': 'NA', 'yaw': 'NA', 'pitch': 'NA', 'roll': 'NA'
}
telemetry_lock = threading.Lock()
notification_queue = Queue()
last_notification_time = 0

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

def send_whatsapp_notification(message):
    """Send WhatsApp message using pywhatkit"""
    try:
        pywhatkit.sendwhatmsg_instantly(
            phone_no=WHATSAPP_NUMBER,
            message=message,
            wait_time=15,
            tab_close=True
        )
        return True
    except Exception as e:
        print(f"Failed to send WhatsApp notification: {e}")
        return False

def notification_worker():
    """Handle notifications in a separate thread"""
    global last_notification_time
    
    while True:
        try:
            # Get the next notification task
            task = notification_queue.get()
            if task is None:  # Exit signal
                break
                
            current_time = time.time()
            
            # Check cooldown period
            if current_time - last_notification_time < NOTIFICATION_COOLDOWN:
                print(f"Notification skipped (cooldown active)")
                continue
                
            # Send notification
            if send_whatsapp_notification(task['message']):
                last_notification_time = current_time
                print("WhatsApp notification sent successfully")
            else:
                print("Failed to send WhatsApp notification")
                
        except Exception as e:
            print(f"Notification error: {e}")

def main(zmq_address="tcp://localhost:5555"):
    # Start telemetry thread
    threading.Thread(target=telemetry_thread, daemon=True).start()
    
    # Start notification thread
    notification_thread = threading.Thread(target=notification_worker, daemon=True)
    notification_thread.start()

    # Initialize YOLO model
    model = YOLO("yolo11n.pt")
    
    # Initialize SORT tracker
    tracker = Sort()
    
    # ZMQ setup
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(zmq_address)
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    sub_socket.setsockopt(zmq.RCVTIMEO, 1000)
    
    # Video writer setup with timestamped filenames
    video_writer = None
    video_filename = f"{TIMESTAMP}.mp4"
    metadata_filename = f"metadata_{TIMESTAMP}.csv"
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    metadata_path = os.path.join(OUTPUT_DIR, metadata_filename)
    
    with open(metadata_path, 'w', newline='') as metadata_file:
        csv_writer = csv.writer(metadata_file)
        csv_writer.writerow(['Frame', 'Timestamp', 'Latitude', 'Longitude', 
                           'Altitude', 'Battery', 'HumanCount', 'Confidence'])
        
        print(f"Detection system connected to ZMQ at {zmq_address}")
        print(f"Saving outputs to: {os.path.abspath(OUTPUT_DIR)}")
        cv2.namedWindow("Human Detection", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        last_human_detection_time = 0
        
        try:
            while True:
                try:
                    # Receive frame
                    jpeg_buffer = sub_socket.recv(flags=zmq.NOBLOCK)
                    frame = cv2.imdecode(np.frombuffer(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                        
                    # Initialize video writer
                    if video_writer is None:
                        frame_size = (frame.shape[1], frame.shape[0])
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, frame_size)
                        print(f"Video recording started: {video_filename}")
                    
                    # Get telemetry
                    with telemetry_lock:
                        tel = latest_telemetry.copy()
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Detection
                    results = model(frame, imgsz=640)[0]
                    detections = []
                    human_count = 0
                    avg_conf = 0
                    
                    for box in results.boxes.data:
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        cls = int(cls)
                        if cls in [0, 1]:  # Only people (0) and bicycles (1)
                            detections.append([x1, y1, x2, y2, conf])
                            if cls == 0:  # Count only humans
                                human_count += 1
                                avg_conf += conf
                    
                    # SORT tracking
                    if len(detections) > 0:
                        detections_array = np.array(detections)
                        tracked_objects = tracker.update(detections_array)
                        avg_conf = avg_conf / human_count if human_count > 0 else 0
                        
                        # Check if we should send notification
                        current_time = time.time()
                        if human_count > 0 and (current_time - last_human_detection_time > NOTIFICATION_COOLDOWN):
                            message = (f"🚨 Human Detection Alert 🚨\n"
                                     f"📍 Location: {tel['lat']}, {tel['lon']}\n"
                                     f"👥 Count: {human_count}\n"
                                     f"🕒 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            notification_queue.put({'message': message})
                            last_human_detection_time = current_time
                    else:
                        tracked_objects = []
                    
                    # Draw results
                    for obj in tracked_objects:
                        x1, y1, x2, y2, obj_id = obj
                        color = (0, 255, 0) if int(obj_id) % 2 == 0 else (0, 0, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"{int(obj_id)}", (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Overlay info
                    info_text = f"Humans: {human_count} | Conf: {avg_conf:.2f}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"{tel['lat']}, {tel['lon']}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Write output
                    video_writer.write(frame)
                    csv_writer.writerow([
                        frame_count, timestamp, 
                        tel['lat'], tel['lon'], tel['alt'],
                        tel['bat'], human_count, avg_conf
                    ])
                    frame_count += 1
                    
                    # Display
                    cv2.imshow("Human Detection", frame)
                    if cv2.waitKey(1) in (ord('q'), 27):
                        break
                        
                except zmq.Again:
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if video_writer is not None:
                video_writer.release()
            sub_socket.close()
            context.term()
            cv2.destroyAllWindows()
            notification_queue.put(None)  # Signal notification thread to exit
            notification_thread.join()
            print(f"Output saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    import sys
    zmq_address = sys.argv[1] if len(sys.argv) > 1 else "tcp://localhost:5555"
    main(zmq_address)
