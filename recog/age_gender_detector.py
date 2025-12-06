# age_gender_detector.py
import cv2
import numpy as np
import os

class AgeGenderDetector:
    """
    Detects age and gender from face crops using pre-trained Caffe models.

    Models from OpenCV's DNN module:
    - Gender model: Binary classification (Male/Female)
    - Age model: 8 age ranges
    """

    def __init__(self, models_dir=None):
        """
        Initialize age and gender detector.

        Args:
            models_dir: Directory containing model files (optional)
        """
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")

        # Age ranges
        self.age_ranges = [
            '(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)'
        ]

        # Gender labels
        self.gender_labels = ['Male', 'Female']

        # Model file paths
        self.gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
        self.gender_model = os.path.join(models_dir, "gender_net.caffemodel")
        self.age_proto = os.path.join(models_dir, "age_deploy.prototxt")
        self.age_model = os.path.join(models_dir, "age_net.caffemodel")

        # Check if models exist
        self.models_available = all([
            os.path.exists(self.gender_proto),
            os.path.exists(self.gender_model),
            os.path.exists(self.age_proto),
            os.path.exists(self.age_model)
        ])

        if self.models_available:
            try:
                # Load networks
                self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
                self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
                print("Age/Gender models loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load age/gender models: {e}")
                self.models_available = False
        else:
            print("Warning: Age/gender model files not found")
            print("Download models from: https://github.com/GilLevi/AgeGenderDeepLearning")
            print(f"Place them in: {models_dir}/")
            self.gender_net = None
            self.age_net = None

    def preprocess_face(self, face_crop):
        """
        Preprocess face crop for age/gender detection.

        Args:
            face_crop: BGR face image

        Returns:
            Preprocessed blob
        """
        # Model was trained on 227x227 images
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        blob = cv2.dnn.blobFromImage(
            face_crop,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )
        return blob

    def detect_gender(self, face_crop):
        """
        Detect gender from face crop.

        Args:
            face_crop: BGR face image

        Returns:
            Tuple of (gender_label, confidence)
        """
        if not self.models_available or self.gender_net is None:
            return "Unknown", 0.0

        try:
            blob = self.preprocess_face(face_crop)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()

            gender_idx = gender_preds[0].argmax()
            gender = self.gender_labels[gender_idx]
            confidence = float(gender_preds[0][gender_idx])

            return gender, confidence
        except Exception as e:
            print(f"Gender detection error: {e}")
            return "Unknown", 0.0

    def detect_age(self, face_crop):
        """
        Detect age range from face crop.

        Args:
            face_crop: BGR face image

        Returns:
            Tuple of (age_range, confidence)
        """
        if not self.models_available or self.age_net is None:
            return "Unknown", 0.0

        try:
            blob = self.preprocess_face(face_crop)
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()

            age_idx = age_preds[0].argmax()
            age_range = self.age_ranges[age_idx]
            confidence = float(age_preds[0][age_idx])

            return age_range, confidence
        except Exception as e:
            print(f"Age detection error: {e}")
            return "Unknown", 0.0

    def detect_age_gender(self, face_crop):
        """
        Detect both age and gender from face crop.

        Args:
            face_crop: BGR face image

        Returns:
            Dictionary with age and gender information
        """
        if face_crop is None or face_crop.size == 0:
            return {
                'age_range': 'Unknown',
                'age_confidence': 0.0,
                'gender': 'Unknown',
                'gender_confidence': 0.0
            }

        gender, gender_conf = self.detect_gender(face_crop)
        age_range, age_conf = self.detect_age(face_crop)

        return {
            'age_range': age_range,
            'age_confidence': age_conf,
            'gender': gender,
            'gender_confidence': gender_conf
        }

    def is_available(self):
        """Check if models are available."""
        return self.models_available


class SimpleAgeGenderEstimator:
    """
    Fallback age/gender estimator using simple heuristics when models aren't available.
    This is a placeholder - not accurate, but demonstrates the integration.
    """

    def __init__(self):
        print("Using simple heuristic-based age/gender estimator (not accurate)")

    def detect_age_gender(self, face_crop):
        """
        Simple heuristic-based estimation (placeholder).

        In reality, you would use:
        - Skin texture analysis
        - Facial feature measurements
        - Hair color/style detection

        Args:
            face_crop: BGR face image

        Returns:
            Dictionary with estimated age and gender
        """
        if face_crop is None or face_crop.size == 0:
            return {
                'age_range': 'Unknown',
                'age_confidence': 0.0,
                'gender': 'Unknown',
                'gender_confidence': 0.0
            }

        # Convert to HSV for basic analysis
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)

        # Very basic heuristic (not accurate - just for demonstration)
        # In production, use proper deep learning models
        h, w = face_crop.shape[:2]

        # Analyze brightness (very rough age indicator)
        brightness = np.mean(hsv[:, :, 2])

        if brightness > 140:
            age_range = "(25-32)"
            age_conf = 0.3
        else:
            age_range = "(38-43)"
            age_conf = 0.3

        # Random gender (not accurate)
        gender = "Unknown"
        gender_conf = 0.0

        return {
            'age_range': age_range,
            'age_confidence': age_conf,
            'gender': gender,
            'gender_confidence': gender_conf,
            'note': 'Heuristic estimation - not accurate. Install proper models for real detection.'
        }

    def is_available(self):
        """Always available but not accurate."""
        return True


def download_age_gender_models():
    """
    Instructions for downloading age/gender models.

    The models are from the work:
    "Age and Gender Classification using Convolutional Neural Networks"
    by Gil Levi and Tal Hassner
    """
    instructions = """
    To enable age and gender detection, download these models:

    1. Visit: https://github.com/GilLevi/AgeGenderDeepLearning

    2. Download the following files:
       - age_deploy.prototxt
       - age_net.caffemodel
       - gender_deploy.prototxt
       - gender_net.caffemodel

    3. Alternative direct links (OpenCV models):
       Age Model:
       https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_net.caffemodel

       Gender Model:
       https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/gender_net.caffemodel

       Prototxt files can be found in OpenCV samples.

    4. Place all files in: recog/models/

    Note: Models are ~44MB total
    """
    return instructions
