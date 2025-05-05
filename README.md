# Advanced Object Detection System

This is a real-time object detection application built with Python, using multiple state-of-the-art models including YOLOv8, SSD MobileNet, and Faster R-CNN. The application provides a user-friendly interface for detecting objects in images, videos, and real-time webcam feeds.

## Features

- Multiple model support (YOLOv8, SSD MobileNet, Faster R-CNN)
- Real-time webcam detection
- Video file processing
- Image file processing
- Recording capability
- Statistics tracking
- Alert system
- Export reports with visualizations
- Customizable settings

## Requirements

- Python 3.12 or later
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python object_detection_app.py
   ```

2. The application window will open with the following features:
   - Model Selection: Choose between YOLOv8, SSD MobileNet, and Faster R-CNN
   - Camera Selection: Select available webcams
   - Control buttons for various functions
   - Statistics display
   - Alert notifications

3. Available functions:
   - Detect Image: Load and process a single image
   - Start Webcam: Begin real-time detection using webcam
   - Start Recording: Record video with detection overlay
   - Upload Video: Process a video file
   - Export Report: Generate statistics and visualizations
   - Settings: Adjust confidence threshold and other parameters

## Directory Structure

- `recordings/`: Saved video recordings
- `exports/`: Exported reports and visualizations
- `models/`: Model files and configurations

## Notes

- The application automatically saves detection history to a SQLite database
- Detection statistics are updated in real-time
- Alerts are triggered based on configurable thresholds
- Reports can be exported in CSV format with accompanying visualizations 