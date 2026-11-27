
import cv2
import numpy as np
from pathlib import Path

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer


BASE_DIR = Path(__file__).resolve().parent.parent
ENCODINGS_DIR = BASE_DIR / "encodings"

# Load registered faces
known_faces = {}

for file in ENCODINGS_DIR.glob("*.npy"):
    known_faces[file.stem] = np.load(str(file))

if not known_faces:
    print("No registered faces found!")
    raise SystemExit

# Initialize AI models
detector = FaceDetector()
recognizer = FaceRecognizer()

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Webcam could not be opened!")
    raise SystemExit

print("Webcam started successfully!")
print("Press Q to exit.")

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read camera frame!")
            break

        # Detect faces
        faces = detector.detect_faces(frame)

        if faces is not None:

            for face in faces:

                # Extract face features
                feature = recognizer.extract_feature(frame, face)

                best_name = "Unknown"
                best_score = -1

                # Compare with registered faces
                for name, known_feature in known_faces.items():

                    similarity = recognizer.recognizer.match(
                        feature,
                        known_feature,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )

                    if similarity > best_score:
                        best_score = similarity
                        best_name = name

                # Identify person
                if best_score >= 0.363:
                    label = best_name
                else:
                    label = "Unknown"

                # Draw bounding box
                x, y, w, h = face[:4].astype(int)

                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # Display webcam
        cv2.imshow("Face Recognition System", frame)

        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed successfully!")
