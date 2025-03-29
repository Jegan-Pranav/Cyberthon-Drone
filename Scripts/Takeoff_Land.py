from pymavlink import mavutil
import time
import tkinter as tk
from tkinter import simpledialog

# Create a popup window to ask for user input
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

# Ask user for takeoff altitude and hover time
altitude = float(simpledialog.askstring("Input", "Enter takeoff altitude (meters):"))
hover_time = float(simpledialog.askstring("Input", "Enter hover time (seconds):"))

# Connect to the drone via UDP
connection = mavutil.mavlink_connection("/dev/ttyUSB1", baud=57600)

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

# Function to take off
def takeoff(altitude):
    print(f"Taking off to {altitude} meters...")
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, altitude
    )
    time.sleep(10)
    print("Takeoff complete!")

# Function to land
def land_drone():
    print("Landing...")
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(10)
    print("Landed.")

# Execute the flight plan
set_mode("GUIDED")
arm_drone()
takeoff(altitude)  # Take off to user-specified height
time.sleep(hover_time)  # Hover for user-specified time
land_drone()

