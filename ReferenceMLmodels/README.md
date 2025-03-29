# Drone-Video-Analytics

This project leverages deep learning models for vehicle and human detection from drone video. It provides a solution for traffic and crowd analysis by detecting objects within a user-defined Region of Interest (ROI). Also be used for Surveillance in borders, Disaster Rescue services using a drone camera in realtime.

models/: Contains pre-trained models for object detection (YOLOv11-based models like yolo11n, yolo11n2, and yolo11st) for detecting vehicles and pedestrians.

sample output/: Sample output from the project, showcasing the detection and tracking results.

oneclass.py: Used to collect input frames and detect and track any single objects like pedestrians, bicycles, cars, vans, trucks, tricycles, buses, and motors from a drone video.

trafficvidroi.py: The main script for traffic and crowd analysis, where users can see a default Region of Interest (ROI) and manually change the ROI in realtime of the video by dragging the four points to track vehicles and people within that region.

Running Traffic and Crowd Analysis
To start the traffic and crowd analysis:

Clone the repository:

`git clone https://github.com/Jegan-Pranav/Drone-Video-Analytics.git`

`cd Drone-Video-Analytics`

Make changes in the trafficvidroi.py to input your video file.

Run the following command:

`python3 trafficvidroi.py`

You can interactively change the ROI by dragging the points on the GUI to define the region of interest.```
