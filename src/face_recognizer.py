
import cv2
from pathlib import Path


class FaceRecognizer:
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "face_recognition_sface_2021dec.onnx"
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(model_path),
            ""
        )

    def extract_feature(self, image, face):
        aligned_face = self.recognizer.alignCrop(image, face)

        feature = self.recognizer.feature(aligned_face)

        return feature

    def compare_faces(self, feature1, feature2, threshold=0.363):
        similarity = self.recognizer.match(
            feature1,
            feature2,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        is_same_person = similarity >= threshold

        return is_same_person, similarity
