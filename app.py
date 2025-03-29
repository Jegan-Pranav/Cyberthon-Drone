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
import traceback
import sys

# Configuration
VIDEO_FOLDER = "video"
LOG_FOLDER = "logs"
SCRIPT_FOLDER = "Scripts"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(SCRIPT_FOLDER, exist_ok=True)

app = Flask(__name__)

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
# Helper Functions
###########################################
def monitor_process_output(proc, model_name, log_file):
    """Monitor process output and write to log file"""
    with open(log_file, 'w') as f:
        while True:
            output = proc.stdout.readline()
            if output == b'' and proc.poll() is not None:
                break
            if output:
                line = output.decode().strip()
                print(f"[{model_name}] {line}")
                f.write(f"{line}\n")
                f.flush()
        
        exit_code = proc.poll()
        print(f"Process {model_name} exited with code {exit_code}")
        if model_name in processes:
            del processes[model_name]

def cleanup_zombies():
    """Clean up any terminated processes"""
    for model in list(processes.keys()):
        if processes[model]['process'].poll() is not None:
            del processes[model]

def get_script_path(model):
    """Get absolute path to script with validation"""
    script_path = os.path.join(os.path.dirname(__file__), SCRIPT_FOLDER, f"{model}.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script {script_path} not found")
    return script_path

###########################################
# Model Process Management
###########################################
@app.route('/start_model', methods=['POST'])
def start_model():
    data = request.get_json()
    model = data.get("model")
    
    if not model:
        return jsonify({"status": "error", "error": "No model specified"}), 400
    
    cleanup_zombies()
    
    if model in processes:
        return jsonify({"status": "already running", "model": model, "pid": processes[model]['process'].pid})
    
    try:
        script_path = get_script_path(model)
        os.chmod(script_path, 0o755)  # Ensure executable permissions
        
        log_file = os.path.join(LOG_FOLDER, f"{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        proc = subprocess.Popen(
            [sys.executable, script_path],  # Use same Python interpreter
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            cwd=os.path.dirname(script_path)
        )
        
        processes[model] = {
            'process': proc,
            'start_time': time.time(),
            'log_file': log_file
        }
        
        threading.Thread(
            target=monitor_process_output,
            args=(proc, model, log_file),
            daemon=True
        ).start()
        
        return jsonify({
            "status": "started",
            "model": model,
            "pid": proc.pid,
            "log_file": log_file
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "model": model
        }), 500

@app.route('/stop_model', methods=['POST'])
def stop_model():
    data = request.get_json()
    model = data.get("model")
    
    if not model:
        return jsonify({"status": "error", "error": "No model specified"}), 400
    
    if model not in processes:
        return jsonify({"status": "not running", "model": model})
    
    try:
        proc = processes[model]['process']
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        del processes[model]
        return jsonify({"status": "stopped", "model": model})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "model": model}), 500

@app.route('/list_models', methods=['GET'])
def list_models():
    """List available models in Scripts directory"""
    try:
        script_dir = os.path.join(os.path.dirname(__file__), SCRIPT_FOLDER)
        scripts = [f[:-3] for f in os.listdir(script_dir) if f.endswith('.py') and os.path.isfile(os.path.join(script_dir, f))]
        return jsonify({
            "status": "success",
            "models": scripts,
            "path": script_dir
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/process_status', methods=['GET'])
def process_status():
    """Get status of all running processes"""
    cleanup_zombies()
    return jsonify({
        "running": list(processes.keys()),
        "details": {
            k: {
                "pid": v['process'].pid,
                "running": v['process'].poll() is None,
                "uptime": time.time() - v['start_time'],
                "log_file": v['log_file']
            }
            for k, v in processes.items()
        }
    })

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
                    if video_writer is not None:
                        video_writer.write(frame)
                    
                    # Save frame metadata
                    frame_metadata = {
                        "timestamp": timestamp_str,
                        "latitude": drone_data['lat'],
                        "longitude": drone_data['lng'],
                        "altitude": drone_data['altitude'],
                        "filename": filename
                    }
                    metadata_file = f"{VIDEO_FOLDER}/metadata_{timestamp}.json"
                    with open(metadata_file, 'a') as f:
                        json.dump(frame_metadata, f)
                        f.write('\n')
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Video processing error: {str(e)}")
                time.sleep(1)
    finally:
        if video_writer is not None:
            video_writer.release()
        pub_socket.close()
        context.term()
        udp_socket.close()

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

@app.route('/get_telemetry', methods=['GET'])
def get_telemetry():
    return jsonify({k: drone_data[k] for k in ['lat', 'lng', 'altitude', 'vertical_speed',
                                             'horizontal_speed', 'yaw', 'pitch', 'roll', 'battery']})

@app.route('/')
def index():
    return render_template('index.html')

@atexit.register
def cleanup_processes():
    for name, proc_info in processes.items():
        try:
            proc = proc_info['process']
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except:
                proc.kill()
        except:
            pass

###########################################
# Main Entry Point
###########################################
if __name__ == '__main__':
    # Start video processing thread
    threading.Thread(target=video_processing_thread, daemon=True).start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
