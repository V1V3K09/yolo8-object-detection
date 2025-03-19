import cv2
import torch
import numpy as np
import pyttsx3
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech
engine = pyttsx3.init()

# Function to detect color
def detect_color(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    avg_color = np.mean(roi, axis=(0, 1))  # Average BGR color
    colors = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "White", "Black"]
    color_ranges = [
        ((0, 0, 100), (100, 100, 255)),  # Red
        ((0, 100, 0), (100, 255, 100)),  # Green
        ((100, 0, 0), (255, 100, 100)),  # Blue
        ((0, 100, 100), (100, 255, 255)),  # Yellow
        ((0, 50, 150), (100, 150, 255)),  # Orange
        ((100, 0, 100), (255, 100, 255)),  # Purple
        ((200, 200, 200), (255, 255, 255)),  # White
        ((0, 0, 0), (50, 50, 50)),  # Black
    ]
    for i, (lower, upper) in enumerate(color_ranges):
        if np.all(avg_color >= lower) and np.all(avg_color <= upper):
            return colors[i]
    return "Unknown"

# Function to estimate depth based on bounding box size
def estimate_depth(x1, y1, x2, y2, frame_width):
    box_size = (x2 - x1) * (y2 - y1)
    max_size = frame_width * frame_width * 0.25  # Approx max size for close objects
    depth_ratio = min(box_size / max_size, 1.0)
    if depth_ratio > 0.8:
        return "very close"
    elif depth_ratio > 0.5:
        return "close"
    elif depth_ratio > 0.3:
        return "moderate distance"
    else:
        return "far"

# Function for text-to-speech output
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(1)  # Use 1 for external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_width = frame.shape[1]
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_name = model.names[int(box.cls[0])]  # Object name

            # Estimate depth
            depth = estimate_depth(x1, y1, x2, y2, frame_width)

            # Detect color
            color = detect_color(frame, x1, y1, x2, y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}, {color}, {depth}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak object details
            speak(f"{class_name}, {color}, {depth}")

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
