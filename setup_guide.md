# AI Interview System — Setup & Run Guide

## Final project structure

```
AI_INTERVIEW_SYSTEM/
│
├── backend/
│   ├── main.py                        ← FastAPI app, all routers registered
│   │
│   ├── routers/
│   │   ├── emotion.py                 ← POST /emotion/analyze-frame|video
│   │   ├── voice.py                   ← POST /voice/analyze
│   │   ├── evaluation.py              ← POST /evaluation/answer|answers-batch
│   │   └── interview.py               ← POST /interview/submit (orchestrator)
│   │
│   ├── services/
│   │   ├── emotion_inference.py       ← CNN wrapper (predict_emotion etc.)
│   │   ├── voice_inference.py         ← LSTM wrapper (predict_voice_score)
│   │   ├── bert_evaluation.py         ← BERT similarity scorer
│   │   └── score_engine.py            ← weighted final score + feedback
│   │
│   └── models/
│       ├── emotion_cnn.h5             ← trained after step 3a
│       ├── voice_lstm.h5              ← trained after step 3b
│       ├── voice_feature_mean.npy     ← saved by train_voice_lstm.py
│       ├── voice_feature_std.npy      ← saved by train_voice_lstm.py
│       └── bert_model/                ← auto-downloaded on first /evaluation call
│
├── datasets/
│   ├── fer2013.csv                    ← FER2013 (place here before step 3a)
│   └── archive__4_.zip               ← RAVDESS voice dataset (place here before step 3b)
│
├── training/
│   ├── train_emotion_cnn.py           ← CNN training script
│   └── train_voice_lstm.py            ← LSTM training script (RAVDESS-aware)
│
├── frontend/
│   ├── index.html                     ← landing page
│   ├── interview.html                 ← live session (webcam + mic)
│   ├── report.html                    ← score report
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── animate.js
│       ├── interview.js               ← set API_BASE here
│       └── report.js
│
├── reports/                           ← (optional) saved session JSON reports
└── requirements.txt
```

---

## Step-by-step setup

### 1. Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

> **GPU users:** Replace `tensorflow` with `tensorflow[and-cuda]` and
> replace the `torch` line with the CUDA build (see comments in requirements.txt).
> Training will be ~10-20x faster.

---

### 2. Place datasets

| File | Location |
|------|----------|
| `fer2013.csv` | `datasets/fer2013.csv` |
| `archive__4_.zip` (RAVDESS) | `datasets/archive__4_.zip` |

---

### 3. Train models

#### 3a — CNN (Emotion, FER2013)
```bash
cd AI_INTERVIEW_SYSTEM
python training/train_emotion_cnn.py
# Output: backend/models/emotion_cnn.h5
# Time: ~30-60 min CPU | ~5-10 min GPU
```

#### 3b — LSTM (Voice, RAVDESS)
```bash
python training/train_voice_lstm.py
# Auto-extracts zip, parses filenames, extracts features for 2880 clips
# Output: backend/models/voice_lstm.h5
#         backend/models/voice_feature_mean.npy
#         backend/models/voice_feature_std.npy
# Time: ~20-40 min CPU | ~5 min GPU
```

#### 3c — BERT (Answer evaluation)
No training needed.  
The `all-MiniLM-L6-v2` model auto-downloads (~90MB) on the **first**
`/evaluation/answer` API call and caches to `backend/models/bert_model/`.  
Pre-download it manually if needed:
```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
m.save("backend/models/bert_model")
```

---

### 4. Run the backend

```bash
cd AI_INTERVIEW_SYSTEM/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs

---

### 5. Run the frontend

```bash
cd AI_INTERVIEW_SYSTEM/frontend
python -m http.server 3000
```

Open http://localhost:3000  
> **Must use localhost** (not `file://`) — browsers block camera/mic on `file://` origins.

Set `API_BASE` in `js/interview.js` to match your backend:
```js
const API_BASE = "http://localhost:8000";
```

---

## API reference

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| `POST` | `/emotion/analyze-frame` | image file | `{emotion, confidence, score}` |
| `POST` | `/emotion/analyze-video` | video file | `{average_score, frame_results, frames_analyzed}` |
| `POST` | `/voice/analyze` | audio file (.wav/.mp3/.webm) | `{confidence_score, pace_score, syllable_rate}` |
| `POST` | `/evaluation/answer` | `{candidate_answer, ideal_answer, keywords?}` | `{similarity_score, keyword_score, final_score}` |
| `POST` | `/evaluation/answers-batch` | `{qa_pairs: [...]}` | `{average_score, results: [...]}` |
| `POST` | `/interview/submit` | video + audio[] + qa_data JSON | full report |

---

## Final score weights

Defined in `backend/services/score_engine.py` — adjust to your needs:

```python
WEIGHTS = {
    "emotion":          0.20,   # CNN facial expression
    "voice_confidence": 0.15,   # LSTM tone/confidence
    "voice_pace":       0.10,   # LSTM speaking pace
    "answer":           0.55,   # BERT answer relevance
}
```

---

## Customising questions

Edit the `QUESTIONS` array in `frontend/js/interview.js`:

```js
const QUESTIONS = [
  {
    text: "Your question here?",
    ideal_answer: "Reference answer for BERT comparison.",
    keywords: ["key", "terms", "expected"]
  },
  // ...
];
```

Or fetch them from a backend endpoint and populate dynamically.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera/mic blocked | Serve frontend over `localhost`, not `file://` |
| BERT model slow first call | Pre-download with the snippet in step 3c |
| `No face detected` error | Ensure good lighting; camera must face you |
| Audio `webm` not accepted | Chrome records `audio/webm` — already handled in `voice.py` |
| CORS error in browser | Check `allow_origins` in `backend/main.py` |
| `voice_feature_mean.npy` not found | Run `train_voice_lstm.py` first — it saves these files |