# Real-Time Object Detection, Tracking, and Facial Analysis

This Python script performs real-time object detection using YOLOv8, tracks detected objects, and analyzes faces for emotion and age estimation. It integrates multiple models for comprehensive scene understanding.

## Features

- **Object Detection**: Uses YOLOv8 to detect objects from the COCO dataset.
- **Object Tracking**: Implements OpenCV trackers (CSRT/KCF) to follow detected objects.
- **Facial Analysis**:
  - Face detection using dlib.
  - Emotion recognition (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
  - Age group estimation (e.g., 0-2, 4-6, 15-20, etc.).
- **Real-Time Webcam Support**: Works with default webcam or video files.

## Prerequisites

- Python 3.8+
- Required files (place in the project directory):
  - `yolov8n.pt` (YOLOv8 model)
  - `coco.names` (COCO class labels)
  - `emotion_model.h5` (Emotion recognition model)
  - `age_model.h5` (Age estimation model)
  - `shape_predictor_68_face_landmarks.dat` (dlib face landmark detector)



1. Install dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python numpy keras ultralytics dlib
   ```

   **Note**: For dlib installation issues on Windows, refer to [dlib's official guide](http://dlib.net/compile.html).

## Usage

1. Place all required files (listed in prerequisites) in the project directory.

2. Run the script:
   ```bash
   python object_detection_tracking.py
   ```

3. Press `q` to exit the application.

### Using a Video File
Modify line 102 in the script:
```python
cap = cv2.VideoCapture("path/to/your/video.mp4")  # Replace 0 with video path
```

## Key Functions

- `detect_objects()`: Detects objects using YOLOv8.
- `track_objects()`: Tracks detected objects using OpenCV trackers.
- `predict_emotion_and_age()`: Predicts emotion and age group for detected faces.

## Notes

- **Tracker Selection**: Switch between `TrackerCSRT` (high accuracy) and `TrackerKCF` (fast) in `create_tracker()`.
- **Debug Outputs**: Emotion/age prediction confidence scores are printed in the console.
- **Performance**: Processing speed depends on hardware. Reduce input resolution if lag occurs.

## Troubleshooting

1. **Missing Files Error**:
   - Ensure all required files are in the project directory.
   - Download links:
     - [YOLOv8 Model](https://github.com/ultralytics/ultralytics)
     - [dlib Shape Predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
     - COCO names, emotion, and age models (custom training or third-party sources).

2. **Model Loading Errors**:
   - Verify Keras model compatibility (custom objects are defined in the script).
   - Ensure `compile=False` is retained when loading models.

3. **Webcam Not Working**:
   - Change `cv2.VideoCapture(0)` to use a different camera index (e.g., `1`).

## Sample Output
- Bounding boxes around detected objects (green).
- Emotion and age labels above detected faces.
- Real-time display in a window titled "Frame".

---
