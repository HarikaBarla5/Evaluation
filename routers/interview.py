"""
Interview router — orchestrates a full session:
  1. Receives interview video + per-question audio + answers
  2. Runs emotion analysis on video
  3. Runs voice analysis on each answer's audio
  4. Runs BERT evaluation on each answer's text vs ideal answer
  5. Combines into final report via score_engine

Endpoints:
  POST /interview/submit - full session submission -> final report
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import json
import cv2
import numpy as np

from services.emotion_inference import predict_emotion_sequence
from services.voice_inference import predict_voice_score
from services.bert_evaluation import evaluate_answers_batch
from services.score_engine import compute_interview_report
from routers.emotion import extract_face

router = APIRouter(prefix="/interview", tags=["Interview"])


@router.post("/submit")
async def submit_interview(
    video: UploadFile = File(..., description="Full interview video"),
    answers_audio: List[UploadFile] = File(..., description="One audio file per question"),
    qa_data: str = Form(..., description="JSON string: list of {candidate_answer, ideal_answer, keywords}"),
    sample_rate: int = Form(15)
):
    """
    qa_data example:
    [
      {"candidate_answer": "...", "ideal_answer": "...", "keywords": ["a","b"]},
      {"candidate_answer": "...", "ideal_answer": "...", "keywords": ["c"]}
    ]
    """
    temp_files = []

    try:
        # ---------------- VALIDATE QA DATA ----------------
        try:
            qa_pairs = json.loads(qa_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="qa_data must be valid JSON")

        if len(qa_pairs) != len(answers_audio):
            raise HTTPException(
                status_code=400,
                detail=f"Mismatch: {len(qa_pairs)} answers vs {len(answers_audio)} audio files"
            )

        # ---------------- EMOTION ANALYSIS (video) ----------------
        video_path = f"/tmp/{video.filename}"
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        temp_files.append(video_path)

        cap = cv2.VideoCapture(video_path)
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

        emotion_results = predict_emotion_sequence(faces)

        # ---------------- VOICE ANALYSIS (per question) ----------------
        voice_results = []
        for audio_file in answers_audio:
            audio_path = f"/tmp/{audio_file.filename}"
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)
            temp_files.append(audio_path)

            voice_results.append(predict_voice_score(audio_path))

        # ---------------- ANSWER EVALUATION (BERT) ----------------
        answer_results = evaluate_answers_batch(qa_pairs)

        # ---------------- FINAL REPORT ----------------
        report = compute_interview_report(emotion_results, voice_results, answer_results)
        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview processing failed: {str(e)}")

    finally:
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)