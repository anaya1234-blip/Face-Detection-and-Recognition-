
import cv2
import numpy as np
from pathlib import Path

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer


BASE_DIR = Path(__file__).resolve().parent.parent
ENCODINGS_DIR = BASE_DIR / "encodings"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load registered faces
known_faces = {}

for file in ENCODINGS_DIR.glob("*.npy"):
    name = file.stem
    known_faces[name] = np.load(str(file))

if not known_faces:
    print("No registered faces found!")
    raise SystemExit

# Load test image
image_path = BASE_DIR / "images" / "unknown_test.jpg"
image = cv2.imread(str(image_path))

if image is None:
    print("Error: Test image not found!")
    raise SystemExit

# Initialize models
detector = FaceDetector()
recognizer = FaceRecognizer()

# Detect faces
faces = detector.detect_faces(image)

if faces is None or len(faces) == 0:
    print("No face detected!")
    raise SystemExit

print("Faces detected:", len(faces))

# Recognize each face
for face in faces:
    feature = recognizer.extract_feature(image, face)

    best_name = "Unknown"
    best_score = -1

    for name, known_feature in known_faces.items():
        similarity = recognizer.recognizer.match(
            feature,
            known_feature,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        if similarity > best_score:
            best_score = similarity
            best_name = name

    if best_score >= 0.363:
        label = best_name
    else:
        label = "Unknown"

    print(f"Person: {label} | Similarity: {best_score:.3f}")

    x, y, w, h = face[:4].astype(int)

    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.putText(
        image,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# Save recognition result
output_path = OUTPUT_DIR / "recognition_result.jpg"
cv2.imwrite(str(output_path), image)

print("Recognition result saved:", output_path)
