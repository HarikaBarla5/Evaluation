"""
Resume router
Endpoints:
  POST /resume/upload  - PDF/DOCX upload -> 5 AI-generated questions
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.resume_service import extract_resume_text, generate_questions

router = APIRouter(prefix="/resume", tags=["Resume"])

ALLOWED_EXT = (".pdf", ".docx", ".doc")
MAX_SIZE_MB = 5


@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    # validate extension
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Please upload PDF or DOCX."
        )

    contents = await file.read()

    # validate size
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_SIZE_MB}MB."
        )

    try:
        resume_text = extract_resume_text(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read resume: {str(e)}")

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Resume appears to be empty or unreadable.")

    try:
        questions = generate_questions(resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

    return {
        "status": "success",
        "filename": file.filename,
        "questions": questions
    }