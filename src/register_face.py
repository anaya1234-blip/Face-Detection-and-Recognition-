
import cv2
import numpy as np
from pathlib import Path

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
ENCODINGS_DIR = BASE_DIR / "encodings"
ENCODINGS_DIR.mkdir(exist_ok=True)

# Get person details
name = input("Enter person's name: ").strip()
image_path = input("Enter image path: ").strip().strip('"')

if not name or not image_path:
    print("Error: Name and image path are required!")
    raise SystemExit

# Prepare safe filename
safe_name = "".join(
    char if char.isalnum() or char in "_-" else "_"
    for char in name
)

# Load image
image_path = Path(image_path)

if not image_path.is_absolute():
    image_path = BASE_DIR / image_path

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Could not load image!")
    raise SystemExit

# Initialize models
detector = FaceDetector()
recognizer = FaceRecognizer()

# Detect face
faces = detector.detect_faces(image)

if faces is None or len(faces) == 0:
    print("Error: No face detected!")
    raise SystemExit

if len(faces) != 1:
    print("Error: Please use an image with exactly one face!")
    raise SystemExit

# Extract face features
feature = recognizer.extract_feature(image, faces[0])

# Save encoding
save_path = ENCODINGS_DIR / f"{safe_name}.npy"
np.save(str(save_path), feature)

print(f"Person '{name}' registered successfully!")
print(f"Face encoding saved at: {save_path}")
