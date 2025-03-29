from flask import Flask, render_template, request, jsonify, Response
import threading
import cv2
import numpy as np
import subprocess
import os
import socket
import pickle
import zmq
import time
from datetime import datetime
import json
import atexit

# Add this near your other constants
VIDEO_FOLDER = "video"
os.makedirs(VIDEO_FOLDER, exist_ok=True)


app = Flask(__name__)

# Add this to both Flask apps:

def zmq_telemetry_listener():
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5556")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    
    while True:
        try:
            data = subscriber.recv_json()
            with frame_lock:  # Use your existing frame lock
                drone_data.update(data)
        except zmq.ZMQError as e:
            print(f"ZMQ error: {e}")
            time.sleep(1)

# Start this thread in your main:
threading.Thread(target=zmq_telemetry_listener, daemon=True).start()



# Global variables
latest_frame = None
frame_lock = threading.Lock()
drone_data = {
    'lat': 0.0, 'lng': 0.0, 'altitude': 0.0, 'vertical_speed': 0.0, 
    'horizontal_speed': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'battery': 100
}
processes = {}

# Fix Permissions-Policy Header
@app.after_request
def set_permissions_header(response):
    response.headers['Permissions-Policy'] = (
        'accelerometer=(), autoplay=(), camera=(), geolocation=(), '
        'microphone=(), midi=(), payment=(), usb=()'
    )
    return response

###########################################
# Model Process Management
###########################################
@app.route('/start_model', methods=['POST'])
def start_model():
    data = request.get_json()
    model = data.get("model")
    
    if model not in processes:
        try:
            script_path = os.path.join(os.getcwd(), "Scripts", f"{model}.py")
            # Critical changes for stability:
            proc = subprocess.Popen(
                ["python3", script_path],
                stdout=open(os.devnull, 'w'),  # Discard stdout
                stderr=open(os.devnull, 'w'),  # Discard stderr
                stdin=subprocess.PIPE,
                close_fds=True,  # Don't inherit file descriptors
                start_new_session=True  # Detach from Flask process
            )
            processes[model] = proc
            return jsonify({"status": "started", "model": model})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e), "model": model})
    else:
        return jsonify({"status": "already running", "model": model})

@app.route('/stop_model', methods=['POST'])
def stop_model():
    data = request.get_json()
    model = data.get("model")
    
    if model in processes:
        try:
            proc = processes[model]
            proc.terminate()  # Try gentle termination first
            try:
                proc.wait(timeout=3)  # Wait for clean exit
            except subprocess.TimeoutExpired:
                proc.kill()  # Force kill if needed
            del processes[model]
            return jsonify({"status": "stopped", "model": model})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e), "model": model})
    else:
        return jsonify({"status": "not running", "model": model})
###########################################
# Video Processing Thread
###########################################

def video_processing_thread():
    global latest_frame, drone_data
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5555")
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('0.0.0.0', 9999))
    udp_socket.settimeout(0.1)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    current_video_file = None
    video_writer = None
    last_frame_time = time.time()
    
    try:
        while True:
            try:
                data, _ = udp_socket.recvfrom(65507)
                buffer = pickle.loads(data)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame.copy()
                    
                    # Publish frame
                    _, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    pub_socket.send(jpeg_buffer.tobytes())
                    
                    # Create new video file every hour or when first starting
                    current_time = time.time()
                    if (video_writer is None) or (current_time - last_frame_time > 3600):
                        if video_writer is not None:
                            video_writer.release()
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{VIDEO_FOLDER}/video_{timestamp}.avi"
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                        last_frame_time = current_time
                    
                    # Add timestamp and location to frame
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    location_str = f"Lat: {drone_data['lat']:.6f}, Lng: {drone_data['lng']:.6f}"
                    
                    cv2.putText(frame, timestamp_str, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, location_str, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Write frame to video file
                    video_writer.write(frame)
                    
                    # Save frame metadata
                    frame_metadata = {
                        "timestamp": timestamp_str,
                        "latitude": drone_data['lat'],
                        "longitude": drone_data['lng'],
                        "altitude": drone_data['altitude'],
                        "filename": filename
                    }
                    with open(f"{VIDEO_FOLDER}/metadata_{timestamp}.json", 'a') as f:
                        json.dump(frame_metadata, f)
                        f.write('\n')  # Newline for each frame
                        
            except socket.timeout:
                continue
    finally:
        if video_writer is not None:
            video_writer.release()
        pub_socket.close()
        context.term()
        udp_socket.close()

# Add cleanup handler to ensure video files are properly closed
@app.route('/cleanup', methods=['POST'])
def cleanup():
    if 'video_writer' in globals():
        video_writer.release()
    return jsonify({'status': 'success'})


###########################################
# Flask Routes
###########################################
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_data', methods=['POST'])
def update_data():
    global drone_data
    data = request.get_json()
    if data:
        drone_data.update(data)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify(drone_data)

def telemetry_subscriber_thread():
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5556")  # Different port than video
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    while True:
        try:
            data = sub_socket.recv_json()
            with frame_lock:  # Reuse your existing frame lock
                drone_data.update(data)
        except zmq.ZMQError as e:
            print(f"Telemetry subscriber error: {e}")
            time.sleep(1)

# Add this before starting your video thread
threading.Thread(target=telemetry_subscriber_thread, daemon=True).start()

@app.route('/get_telemetry', methods=['GET'])
def get_telemetry():
    return jsonify({k: drone_data[k] for k in ['lat', 'lng', 'altitude', 'vertical_speed', 
                                             'horizontal_speed', 'yaw', 'pitch', 'roll', 'battery']})

@app.route('/')
def index():
    return render_template('mission.html')

@atexit.register
def cleanup_processes():
    for name, proc in processes.items():
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except:
            pass

###########################################
# Main Entry Point
###########################################
if __name__ == '__main__':
    threading.Thread(target=video_processing_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5002, threaded=True, debug=False, use_reloader=False)
