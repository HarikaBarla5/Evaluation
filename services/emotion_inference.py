"""
Emotion inference utility — used by backend/routers/emotion.py
Maps face image -> emotion label + confidence score (0-100) for interview scoring.
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "backend/models/emotion_cnn.h5"
IMG_SIZE = 48

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Weight each emotion's contribution to an "interview confidence score"
# Tune these based on what behaviors you consider positive in an interview
EMOTION_SCORE_WEIGHTS = {
    "Happy": 90,
    "Neutral": 80,
    "Surprise": 60,
    "Sad": 40,
    "Fear": 30,
    "Angry": 25,
    "Disgust": 20,
}

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def preprocess_face(face_img):
    """face_img: cropped grayscale face, any size -> normalized 48x48x1"""
    face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)
    return face


def predict_emotion(face_img):
    """
    Returns:
        emotion_label (str), confidence (float 0-1), interview_score (float 0-100)
    """
    model = get_model()
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = EMOTION_LABELS[idx]
    confidence = float(preds[idx])

    interview_score = EMOTION_SCORE_WEIGHTS[label] * confidence
    return label, confidence, interview_score


def predict_emotion_sequence(face_imgs):
    """
    For a list of frames (e.g. from interview video), returns average interview score
    and a breakdown of detected emotions over time.
    """
    results = []
    for img in face_imgs:
        label, conf, score = predict_emotion(img)
        results.append({"emotion": label, "confidence": conf, "score": score})

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    return {
        "average_score": round(avg_score, 2),
        "frame_results": results
    }