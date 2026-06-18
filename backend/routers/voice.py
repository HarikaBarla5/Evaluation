"""
Voice analysis router
Endpoints:
  POST /voice/analyze - audio file -> confidence + pace scores
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.voice_inference import predict_voice_score

router = APIRouter(prefix="/voice", tags=["Voice"])

ALLOWED_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


@router.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXT}")

    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = predict_voice_score(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
