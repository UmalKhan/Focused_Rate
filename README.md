# Focused_Rate

## Overview
This project develops a real-time computer vision and data analytics system to measure and visualize audience engagement by determining if individuals are focused on a specific point (e.g., a screen, presentation, or speaker). It leverages advanced head pose estimation to infer gaze direction and then aggregates this data to provide valuable insights into attention levels across different courses, lecturers, and classes.

## Features
Real-time Gaze Estimation: Detects faces and estimates 3D head pose to infer gaze direction towards a defined target point.
Multi-Person Detection: Capable of tracking engagement for multiple individuals simultaneously.
Data Collection: Records timestamps, individual face IDs, engagement status ("Looking at Target" or "Looking Away"), and gaze angles into a structured CSV file.
Data Aggregation: Processes raw engagement data to calculate average focused rates by:
Course
Lecturer
Class
Intuitive Visualization: Generates clear and aesthetically pleasing bar charts to visualize engagement metrics, making complex data easily understandable.
Modular Design: Separates the real-time detection, data collection, and data analysis/visualization components for maintainability and scalability.
How It Works
Camera Calibration: A one-time process to understand the camera's intrinsic properties, crucial for accurate 3D pose estimation.
Real-time Detection: Uses MediaPipe's Face Mesh to detect 3D facial landmarks from a live video feed.
3D Head Pose Estimation: Employs OpenCV's Perspective-n-Point (PnP) algorithm to derive the head's 3D rotation and translation relative to the camera.
Gaze Direction Calculation: Infers the gaze vector from the head's orientation and compares it to a predefined 3D target point.
Data Logging: Records the engagement status and gaze angle for each detected face per frame into a CSV file.
Offline Analysis: A separate script reads the collected data, calculates average focus rates, and generates visualizations to provide insights into engagement patterns.

## Technologies Used
Python: The primary programming language.
MediaPipe: For robust and efficient real-time facial landmark detection.
OpenCV (cv2): For camera calibration, PnP algorithm, and general image/video processing.
NumPy: For numerical operations, especially with 3D vectors and matrices.
Pandas: For powerful data loading, manipulation, and aggregation.
Matplotlib: For creating static, high-quality visualizations.
Seaborn: Built on Matplotlib, for enhanced aesthetics and simpler statistical plotting.
Getting Started
Follow these steps to set up and run the project locally.

## Prerequisites
Python 3.7+
Webcam (for live detection)
A printed chessboard pattern (for camera calibration)

## Installation
Install the required libraries:
pip install opencv-python mediapipe numpy pandas matplotlib seaborn

## Step 1: Camera Calibration
This is a crucial one-time step for accurate 3D gaze estimation.
Prepare Chessboard: Print a chessboard pattern (e.g., 9x6 inner corners) on matte paper. Measure the exact size of one square (e.g., 20mm). Update the CHECKERBOARD and SQUARE_SIZE variables in calibrate_camera.py accordingly.
Run the calibration script:
python calibrating.py
Capture Images: Choose 'c' to capture new images. Hold the chessboard steady in various positions, angles, and distances within the webcam's view. Press 's' to save images and 'q' to quit. Aim for at least 10-20 successfully detected images.
Process Images: The script will automatically process the captured images and perform calibration.
Save Results: The camera_matrix and dist_coeffs will be saved to camera_calibration.npz. You will need these for the real-time detection script.

## Step 2: Run Real-time Gaze Detection and Data Collection
Define Target Point: In your check.py (the script that logs data), you'll need to define the TARGET_POINT_3D coordinates. This is the 3D (X, Y, Z) location of the point you want to monitor focus on, relative to your camera's lens.
Example: TARGET_POINT_3D = np.array([[-500.0, 0.0, 5000.0]]) (5 meters in front, 0.5 meters to the left, at camera height, all in mm).
Update Calibration in Main Script: Load the camera_matrix and dist_coeffs from camera_calibration.npz into your main_gaze_detector.py script.
Run the main detection script:
python main_gaze_detector.py
Collect Data: The script will open your webcam feed. As it detects faces and estimates gaze, it will log data to focus_data.csv. Press 'q' to stop the detection and close the CSV file.

## Step 3: Analyze and Visualize Data
Ensure data is collected: Make sure students.csv exists and contains the collected engagement data.
Run the analysis script:
python show.py
View Plots: The script will display the generated bar charts showing average focused rates by course, lecturer, and class.
