from flask import Flask, render_template, send_file, Response
import os
import mimetypes
from datetime import datetime
import subprocess
import platform
import webbrowser

app = Flask(__name__)

# Configuration
app.config['VIDEO_FOLDER'] = 'video'
app.config['ALLOWED_EXTENSIONS'] = {'.mp4', '.webm', '.ogg', '.avi'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB upload limit

def get_video_files():
    """Get all video files from the video folder with metadata"""
    videos = []
    for filename in os.listdir(app.config['VIDEO_FOLDER']):
        filepath = os.path.join(app.config['VIDEO_FOLDER'], filename)
        if os.path.isfile(filepath):
            name, ext = os.path.splitext(filename)
            if ext.lower() in app.config['ALLOWED_EXTENSIONS']:
                stat = os.stat(filepath)
                videos.append({
                    'filename': filename,
                    'name': name,
                    'size': round(stat.st_size / (1024 * 1024), 2),  # MB
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'filepath': filepath  # Add full filepath for VLC
                })
    return sorted(videos, key=lambda x: x['modified'], reverse=True)

def open_in_vlc(filepath):
    """Open video file in VLC player"""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', '-a', 'VLC', filepath])
        else:  # Linux and others
            subprocess.run(['vlc', filepath])
        return True
    except Exception as e:
        print(f"Error opening VLC: {e}")
        return False

@app.route('/')
def index():
    """Main page showing all videos"""
    videos = get_video_files()
    return render_template('dashboard.html', videos=videos)

@app.route('/play/<filename>')
def play(filename):
    """Open video in VLC player"""
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "Video not found", 404
    
    if open_in_vlc(video_path):
        return "Opening in VLC..."
    else:
        return "Failed to open VLC player", 500

@app.route('/stream/<filename>')
def stream(filename):
    """Stream video with support for byte-range requests"""
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    if not os.path.exists(video_path):
        return "Video not found", 404
    return send_file(video_path, mimetype=mimetypes.guess_type(video_path)[0], conditional=True)

@app.route('/convert/<filename>')
def convert(filename):
    """Convert video to MP4 format using ffmpeg"""
    input_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    output_filename = f"converted_{os.path.splitext(filename)[0]}.mp4"
    output_path = os.path.join(app.config['VIDEO_FOLDER'], output_filename)
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'aac', '-f', 'mp4',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)
    mimetypes.init()
    app.run(host='0.0.0.0', port=5001, debug=True)
