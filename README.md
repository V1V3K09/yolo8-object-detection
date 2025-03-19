# Object Detection for Blind using YOLOv8

This project uses YOLOv8 for real-time object detection, color recognition, and verbal depth feedback to assist visually impaired individuals. The model detects objects, estimates depth, and announces details via text-to-speech.

## Features
- Object detection using YOLOv8
- Color detection of detected objects
- Verbal feedback using pyttsx3
- Depth estimation (placeholder, can be integrated with a stereo camera)
  

## Environment Setup
conda create --name yolo8 python=3.9 -y

## Environment Activation
conda activate yolo8
## Installation
### Prerequisites
Ensure Python (>=3.8) is installed. Then, install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the script to start object detection:
```sh
python object_detection_blind.py
```
Press `q` to quit the application.

## Troubleshooting
- If `ultralytics` is not found, install it separately:
  ```sh
  pip install ultralytics --no-cache-dir
  ```
- Ensure the correct Python environment is activated before running the script.

## Future Enhancements
- Integrate depth sensors for accurate distance measurement.
- Improve object tracking for better assistance.

