import cv2
import numpy as np
from keras.models import load_model
import keras.metrics as metrics
import keras.losses as losses
import dlib
import os
from ultralytics import YOLO

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Check if all required files exist
required_files = [
    "yolov8n.pt",  # YOLOv8 model
    "coco.names",
    "emotion_model.h5",
    "age_model.h5",
    "shape_predictor_68_face_landmarks.dat"
]

for file in required_files:
    file_path = os.path.join(base_dir, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

# Load YOLOv8
model = YOLO(os.path.join(base_dir, "yolov8n.pt"))

# Load class names
with open(os.path.join(base_dir, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define custom objects with metrics and losses
custom_objects = {
    'mse': losses.mean_squared_error,
    'mean_squared_error': losses.mean_squared_error,
    'MSE': losses.mean_squared_error,  # Changed from metrics.mean_squared_error
    'accuracy': 'accuracy'  # Add basic accuracy metric
}

# Load emotion recognition model
try:
    emotion_model = load_model(os.path.join(base_dir, "emotion_model.h5"), 
                             custom_objects=custom_objects,
                             compile=False)  # Add compile=False
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error loading emotion model: {e}")
    raise

# Load age estimation model
try:
    age_model = load_model(os.path.join(base_dir, "age_model.h5"), 
                          custom_objects=custom_objects,
                          compile=False)  # Add compile=False
    age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error loading age model: {e}")
    raise

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat"))

# Define emotion and age labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
age_labels = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

def detect_objects(frame, model):
    results = model(frame)
    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
        confidences.extend(result.boxes.conf.cpu().numpy())
        class_ids.extend(result.boxes.cls.cpu().numpy().astype(int))

    return class_ids, confidences, boxes

def create_tracker():
    # You can choose between different trackers
    tracker = cv2.TrackerCSRT_create()
    # tracker = cv2.TrackerKCF_create()
    return tracker

def track_objects(frame, tracker, bbox):
    ok, bbox = tracker.update(frame)
    return ok, bbox

def predict_emotion_and_age(frame, face_bbox):
    x, y, w, h = face_bbox
    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return "Unknown", "Unknown"

    # Convert face to grayscale for emotion prediction
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Resize face for emotion prediction
    face_resized = cv2.resize(face_gray, (48, 48))
    face_resized = face_resized.astype("float") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)
    face_resized = np.expand_dims(face_resized, axis=-1)

    # Predict emotion
    emotion_pred = emotion_model.predict(face_resized)
    emotion_label = emotion_labels[np.argmax(emotion_pred)]
    print(f"Emotion prediction: {emotion_pred}, Label: {emotion_label}")  # Debug statement

    # Resize face for age prediction
    face_resized_age = cv2.resize(face, (256, 256))  # Change to (256, 256)
    face_resized_age = face_resized_age.astype("float") / 255.0
    face_resized_age = np.expand_dims(face_resized_age, axis=0)

    # Predict age
    age_pred = age_model.predict(face_resized_age)
    age_label = age_labels[np.argmax(age_pred)]
    print(f"Age prediction: {age_pred}, Label: {age_label}")  # Debug statement

    return emotion_label, age_label

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

    trackers = []
    bboxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(trackers) == 0:
            # Perform object detection
            class_ids, confidences, boxes = detect_objects(frame, model)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = create_tracker()
                tracker.init(frame, bbox)
                trackers.append(tracker)
                bboxes.append(bbox)

        else:
            # Track the objects
            new_bboxes = []
            for i, tracker in enumerate(trackers):
                ok, bbox = track_objects(frame, tracker, bboxes[i])

                if ok:
                    # Draw bounding box
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

                    # Detect faces within the bounding box
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray, 0)

                    for face in faces:
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        face_bbox = (x, y, w, h)
                        print(f"Detected face bbox: {face_bbox}")  # Debug statement

                        # Predict emotion and age
                        emotion_label, age_label = predict_emotion_and_age(frame, face_bbox)

                        # Draw emotion and age labels
                        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Age: {age_label}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    new_bboxes.append(bbox)
                else:
                    # If tracking fails, remove the tracker
                    del trackers[i]
                    del bboxes[i]
                    break

            bboxes = new_bboxes

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()