"""
Emotion analysis router
Endpoints:
  POST /emotion/analyze-frame   - single image -> emotion + score
  POST /emotion/analyze-video   - video file -> aggregated interview score
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.emotion_inference import predict_emotion, predict_emotion_sequence

router = APIRouter(prefix="/emotion", tags=["Emotion"])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_face(image_bgr):
    """Detect and crop the largest face from a BGR image. Returns grayscale face or None."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        return None

    # pick largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y:y + h, x:x + w]


@router.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    face = extract_face(image)
    if face is None:
        raise HTTPException(status_code=422, detail="No face detected in frame")

    label, confidence, score = predict_emotion(face)
    return {
        "emotion": label,
        "confidence": round(confidence, 4),
        "score": round(score, 2)
    }


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 15):
    """
    sample_rate: analyze every Nth frame (default every 15th frame, ~2 fps for 30fps video)
    """
    contents = await file.read()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")

    faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            face = extract_face(frame)
            if face is not None:
                faces.append(face)

        frame_idx += 1

    cap.release()

    if not faces:
        raise HTTPException(status_code=422, detail="No faces detected in video")

    result = predict_emotion_sequence(faces)
    result["frames_analyzed"] = len(faces)
    return result
