
import cv2
from pathlib import Path


class FaceDetector:
    def __init__(self):
        self.model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "face_detection_yunet_2023mar.onnx"
        )

        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
               0.5,
               0.3,
               5000
        )

    def detect_faces(self, image):
        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        return faces

    def draw_faces(self, image, faces):
        if faces is None:
            return image

        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return image
