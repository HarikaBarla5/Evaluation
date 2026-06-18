from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import emotion, voice, evaluation, interview, resume
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="AI Interview System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(emotion.router)
app.include_router(voice.router)
app.include_router(evaluation.router)
app.include_router(interview.router)
app.include_router(resume.router)


@app.get("/")
def root():
    return {"status": "AI Interview backend running"}