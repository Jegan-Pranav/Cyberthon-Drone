#!/bin/bash
echo "Stopping all services..."
pkill -f "dashboard.py|mission.py|send_coordinates.py" >/dev/null 2>&1
fuser -k 5000/tcp >/dev/null 2>&1
echo "Services stopped"
