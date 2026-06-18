"""
AI Interview Evaluation Pipeline
Exported from the notebook for use in the Flask backend.
"""
import os, re, json
import numpy as np
import spacy
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
from tensorflow.keras import layers, models
from fer import FER
import cv2
import librosa

nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
facial_emotion_detector = FER(mtcnn=True)

SKILL_DB = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "nosql",
    "mongodb", "mysql", "postgresql", "html", "css", "react", "angular", "vue",
    "node.js", "django", "flask", "fastapi", "spring boot",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy",
    "cnn", "rnn", "lstm", "bert", "transformer", "data structures",
    "algorithms", "aws", "azure", "gcp", "docker", "kubernetes", "git",
    "rest api", "microservices", "data analysis", "data visualization",
    "tableau", "power bi", "excel", "linux", "agile", "scrum"
]

QUESTION_BANK = {
    "python": [("What is the difference between a list and a tuple in Python?",
        "Lists are mutable and can be changed after creation, while tuples are immutable and cannot be modified once created.")],
    "machine learning": [("Explain the difference between supervised and unsupervised learning.",
        "Supervised learning uses labeled data to train a model to predict outcomes, while unsupervised learning finds patterns or groupings in unlabeled data.")],
    "cnn": [("What is the role of a convolutional layer in a CNN?",
        "A convolutional layer applies filters/kernels across the input to extract spatial features like edges, textures and patterns, reducing the need for manual feature engineering.")],
    "lstm": [("Why are LSTMs preferred over simple RNNs for sequence data?",
        "LSTMs use gating mechanisms to retain long-term dependencies and avoid the vanishing gradient problem that affects simple RNNs.")],
    "bert": [("What makes BERT different from traditional word embedding models?",
        "BERT is a bidirectional transformer model that generates context-aware embeddings, unlike static embeddings like Word2Vec.")],
    "sql": [("What is the difference between INNER JOIN and LEFT JOIN?",
        "INNER JOIN returns only matching rows from both tables, while LEFT JOIN returns all rows from the left table and matching rows from the right.")],
    "flask": [("How do you create a simple route in Flask?",
        "You use the @app.route decorator on a function, specifying the URL path, and the function returns the response for that route.")],
    "docker": [("What is the difference between a Docker image and a Docker container?",
        "A Docker image is a static, read-only template, while a container is a running instance of that image.")],
    "rest api": [("What are the main HTTP methods used in REST APIs and their purpose?",
        "GET retrieves data, POST creates new data, PUT updates existing data, and DELETE removes data.")],
}
DEFAULT_QUESTIONS = [("Describe a challenging technical problem you solved recently and your approach.",
    "A good answer describes the problem, the approach/methodology, tools used, and the final outcome.")]

VOICE_EMOTION_CLASSES = ["calm", "happy", "sad", "angry", "fearful", "nervous", "confident"]

HR_QUESTION_BANK = [
    {"question": "Tell me about yourself.",
     "ideal_answer": "A concise summary of professional background, key skills, relevant experience, and career goals aligned with the role."},
    {"question": "Why do you want to work with us?",
     "ideal_answer": "A genuine, specific reason connecting the company's mission, values, or projects to the candidate's own career goals and skills."},
    {"question": "Describe a time you handled conflict in a team.",
     "ideal_answer": "A structured example (situation, action, result) showing communication, empathy, and a positive resolution to a team conflict."},
    {"question": "Where do you see yourself in 5 years?",
     "ideal_answer": "A realistic growth plan showing ambition, alignment with the company's career path, and continued skill development."},
]

def extract_text_from_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    elif ext == ".docx":
        d = docx.Document(file_path)
        text = "\n".join(p.text for p in d.paragraphs)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported resume format: {ext}")
    return text

def extract_skills(text, skill_db=SKILL_DB):
    text_lower = text.lower()
    found = set()
    for skill in skill_db:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)

def resume_jd_match_score(resume_text, jd_text):
    emb = bert_model.encode([resume_text, jd_text], convert_to_tensor=True)
    return round(util.cos_sim(emb[0], emb[1]).item() * 100, 2)

def score_answer_bert(candidate_answer, ideal_answer):
    if not candidate_answer.strip():
        return 0.0
    emb = bert_model.encode([candidate_answer, ideal_answer], convert_to_tensor=True)
    return round(max(util.cos_sim(emb[0], emb[1]).item(), 0) * 100, 2)

def generate_written_test(skills, num_questions=5):
    questions = []
    for skill in skills:
        if skill in QUESTION_BANK:
            for q, a in QUESTION_BANK[skill]:
                questions.append({"skill": skill, "question": q, "ideal_answer": a})
        if len(questions) >= num_questions:
            break
    while len(questions) < num_questions:
        q, a = DEFAULT_QUESTIONS[0]
        questions.append({"skill": "general", "question": q, "ideal_answer": a})
    return questions[:num_questions]

def extract_voice_features(audio_path, max_pad_len=174, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, pad_width=((0,0),(0, max_pad_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.T

VOICE_MODEL_PATH = "voice_emotion_lstm.h5"
voice_lstm_model = tf.keras.models.load_model(VOICE_MODEL_PATH) if os.path.exists(VOICE_MODEL_PATH) else None

def analyze_voice_emotion(audio_path):
    features = extract_voice_features(audio_path)
    if voice_lstm_model is not None:
        pred = voice_lstm_model.predict(features[np.newaxis, ...], verbose=0)[0]
        probs = {c: round(float(p), 3) for c, p in zip(VOICE_EMOTION_CLASSES, pred)}
    else:
        rms = float(np.mean(librosa.feature.rms(y=librosa.load(audio_path, sr=22050)[0])))
        confidence = min(1.0, rms * 20)
        probs = {c: round((confidence if c in ["calm","confident"] else (1-confidence)/5), 3) for c in VOICE_EMOTION_CLASSES}
    return {"voice_emotion_profile": probs, "dominant_voice_emotion": max(probs, key=probs.get)}

def analyze_facial_emotions(video_path, sample_every_n_frames=15):
    cap = cv2.VideoCapture(video_path)
    emotion_totals, frame_count, sampled = {}, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % sample_every_n_frames == 0:
            result = facial_emotion_detector.detect_emotions(frame)
            if result:
                for emo, score in result[0]["emotions"].items():
                    emotion_totals[emo] = emotion_totals.get(emo, 0) + score
                sampled += 1
        frame_count += 1
    cap.release()
    if sampled == 0:
        return {"error": "No face detected in video", "emotion_profile": {}}
    avg = {k: round(v/sampled, 3) for k, v in emotion_totals.items()}
    return {"emotion_profile": avg, "dominant_emotion": max(avg, key=avg.get), "frames_analyzed": sampled}

def evaluate_hr_answers_bert(hr_answers):
    results, total = [], 0
    for q, ans in zip(HR_QUESTION_BANK, hr_answers):
        score = score_answer_bert(ans, q["ideal_answer"])
        total += score
        results.append({"question": q["question"], "answer": ans, "score": score})
    avg = round(total/len(HR_QUESTION_BANK), 2) if HR_QUESTION_BANK else 0
    return {"per_question_scores": results, "overall_text_score": avg}


class Round1ResumeEvaluator:
    THRESHOLD = 50.0
    def evaluate(self, resume_path, job_description):
        text = extract_text_from_resume(resume_path)
        skills = extract_skills(text)
        score = resume_jd_match_score(text, job_description)
        passed = score >= self.THRESHOLD
        return {"round":1, "extracted_skills": skills, "resume_jd_match_score": score,
                "passed": passed, "verdict": "Advance to Round 2" if passed else "Rejected after Round 1"}


class Round2WrittenTestEvaluator:
    THRESHOLD = 50.0
    def generate_test(self, skills, num_questions=5):
        return generate_written_test(skills, num_questions)
    def evaluate(self, test_questions, candidate_answers):
        per_question, total = [], 0
        for q, ans in zip(test_questions, candidate_answers):
            score = score_answer_bert(ans, q["ideal_answer"])
            total += score
            per_question.append({"skill": q["skill"], "question": q["question"], "candidate_answer": ans, "score": score})
        avg = round(total/len(test_questions), 2) if test_questions else 0
        passed = avg >= self.THRESHOLD
        return {"round":2, "per_question_scores": per_question, "overall_score": avg,
                "passed": passed, "verdict": "Advance to Round 3" if passed else "Rejected after Round 2"}


class Round3HREvaluator:
    THRESHOLD = 55.0
    WEIGHTS = {"facial": 0.3, "voice": 0.3, "text": 0.4}
    POSITIVE_FACIAL = ["happy","neutral","surprise"]
    POSITIVE_VOICE = ["calm","confident","happy"]

    def _facial_score(self, r):
        p = r.get("emotion_profile", {})
        return round(min(sum(p.get(e,0) for e in self.POSITIVE_FACIAL),1.0)*100,2) if p else 0.0
    def _voice_score(self, r):
        p = r.get("voice_emotion_profile", {})
        return round(min(sum(p.get(e,0) for e in self.POSITIVE_VOICE),1.0)*100,2) if p else 0.0

    def evaluate(self, video_path, audio_path, hr_answers):
        facial = analyze_facial_emotions(video_path)
        voice = analyze_voice_emotion(audio_path)
        text = evaluate_hr_answers_bert(hr_answers)
        fs, vs, ts = self._facial_score(facial), self._voice_score(voice), text["overall_text_score"]
        final = round(fs*self.WEIGHTS["facial"] + vs*self.WEIGHTS["voice"] + ts*self.WEIGHTS["text"], 2)
        passed = final >= self.THRESHOLD
        return {"round":3, "facial_analysis": facial, "voice_analysis": voice, "text_analysis": text,
                "component_scores": {"facial_score": fs, "voice_score": vs, "text_score": ts},
                "final_hr_score": final, "passed": passed,
                "verdict": "Candidate Selected" if passed else "Candidate Not Selected"}


class InterviewPipeline:
    def __init__(self):
        self.r1 = Round1ResumeEvaluator()
        self.r2 = Round2WrittenTestEvaluator()
        self.r3 = Round3HREvaluator()
