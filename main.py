
import cv2

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer

# Initialize both models
detector = FaceDetector()
recognizer = FaceRecognizer()

print("YuNet Face Detector loaded successfully!")
print("SFace Face Recognizer loaded successfully!")

# Load test image
image_path = "images/test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Test image not found!")

else:
    # Detect faces
    faces = detector.detect_faces(image)

    if faces is None:
        print("No face detected!")

    else:
        print("Faces detected:", len(faces))

        # Extract face features
        for i, face in enumerate(faces):
            feature = recognizer.extract_feature(image, face)

            print(f"Face {i + 1} features extracted successfully!")
            print("Feature shape:", feature.shape)

    # Draw bounding boxes
    result = detector.draw_faces(image, faces)

    # Save output
    cv2.imwrite("output/face_detection_result.jpg", result)

    print("Result saved in output folder.")
