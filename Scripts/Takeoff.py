from pymavlink import mavutil
import time
import tkinter as tk
from tkinter import simpledialog
import os

FLAG_FILE = "land_now.txt"  # File used to signal landing

# Create a popup window to ask for user input
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

# Ask user for takeoff altitude
altitude = float(simpledialog.askstring("Input", "Enter takeoff altitude (meters):"))

# Connect to the drone via UDP
connection = mavutil.mavlink_connection("/dev/ttyUSB0", baud=57600)

# Wait for the heartbeat
connection.wait_heartbeat()
print("Heartbeat received, drone is ready!")

# Function to change flight mode
def set_mode(mode):
    connection.set_mode(mode)
    print(f"Mode changed to {mode}")
    time.sleep(1)

# Function to arm the drone
def arm_drone():
    print("Arming drone...")
    connection.arducopter_arm()
    time.sleep(4)
    print("Drone armed!")

# Function to take off and hold position
def takeoff_and_hold(altitude):
    print(f"Taking off to {altitude} meters...")
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, altitude
    )
    time.sleep(10)
    print("Takeoff complete! Holding position...")

# Execute the flight plan
set_mode("GUIDED")
arm_drone()
takeoff_and_hold(altitude)  # Take off and hold at the specified height

# Hold position indefinitely but check for landing signal
print("Holding position... Waiting for landing signal.")
while True:
    if os.path.exists(FLAG_FILE):  # If the land script creates this file
        print("Landing signal received! Exiting takeoff script...")
        break
    time.sleep(1)

print("Takeoff script stopped. Run the land script now.")

