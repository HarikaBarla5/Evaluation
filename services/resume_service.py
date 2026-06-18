"""
Resume service — extracts text from PDF/DOCX and generates
5 interview questions using Groq API (FREE, no protobuf conflicts).
"""

import os
import io
import json
import pdfplumber
import docx

# ── Load .env automatically ──────────────────────────────────
try:
    from dotenv import load_dotenv
    _dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(_dir, ".env"))
    load_dotenv(dotenv_path=os.path.join(_dir, "..", ".env"))
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    raise RuntimeError(
        "\n\n❌  GROQ_API_KEY is not set!\n"
        "Add this to backend/.env:\n"
        "    GROQ_API_KEY=gsk_your-key-here\n"
        "Get your FREE key at: https://console.groq.com\n"
    )


# ================= TEXT EXTRACTION =================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ================= QUESTION GENERATION =================

SYSTEM_PROMPT = """You are an expert technical interviewer.
Given a candidate's resume, generate exactly 5 interview questions
tailored specifically to their skills, projects, and experience.

Rules:
- Questions must be specific to THIS candidate — not generic
- Mix technical questions (from their tech stack) with behavioural questions (from their projects)
- Each question must have a detailed ideal answer and 3-5 relevant keywords

Respond ONLY with a valid JSON array. No markdown, no explanation, just raw JSON:
[
  {
    "question": "...",
    "ideal_answer": "...",
    "keywords": ["...", "...", "..."]
  }
]"""


def generate_questions(resume_text: str) -> list:
    # lazy import — keeps groq away from tensorflow startup
    from groq import Groq

    if not resume_text.strip():
        raise ValueError("Resume text is empty — could not extract content")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # free, fast, very capable
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Here is the candidate's resume:\n\n{resume_text[:3000]}\n\nGenerate 5 interview questions."}
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content.strip()

    # strip markdown fences if model adds them
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw: {raw[:300]}")

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Model did not return a valid question list")

    for q in questions:
        if not all(k in q for k in ("question", "ideal_answer", "keywords")):
            raise ValueError(f"Malformed question object: {q}")

    return questions[:5]