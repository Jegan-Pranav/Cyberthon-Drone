#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def find_qgc():
    # Common QGC installation locations in Downloads
    downloads_dir = Path.home() / "Downloads"
    possible_paths = [
        downloads_dir / "QGroundControl.AppImage",
        downloads_dir / "QGroundControl" / "QGroundControl.AppImage",
        downloads_dir / "QGroundControl-x86_64.AppImage"
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def launch_qgc(qgc_path):
    try:
        # Make the AppImage executable if it isn't already
        os.chmod(qgc_path, 0o755)
        
        # Launch QGC
        subprocess.Popen([str(qgc_path)])
        print(f"Successfully launched QGroundControl from {qgc_path}")
    except Exception as e:
        print(f"Error launching QGroundControl: {e}")

def main():
    print("Attempting to launch QGroundControl...")
    
    qgc_path = find_qgc()
    
    if qgc_path:
        launch_qgc(qgc_path)
    else:
        print("QGroundControl not found in Downloads directory.")
        print("Please ensure QGroundControl.AppImage is in your ~/Downloads folder")

if __name__ == "__main__":
    main()

