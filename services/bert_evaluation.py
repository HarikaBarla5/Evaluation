"""
Answer evaluation using BERT sentence embeddings + cosine similarity
against ideal/reference answers.

Uses sentence-transformers (BERT-based) for semantic similarity scoring.
Model dir backend/models/bert_model/ should contain a sentence-transformers
model (e.g. 'all-MiniLM-L6-v2' or 'bert-base-nli-mean-tokens') saved locally.

If empty, the model will be downloaded on first run and cached there.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = "backend/models/bert_model"
MODEL_NAME = "all-MiniLM-L6-v2"  # good balance of speed/accuracy for sentence similarity

_model = None


def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            _model = SentenceTransformer(MODEL_DIR)
        else:
            _model = SentenceTransformer(MODEL_NAME)
            os.makedirs(MODEL_DIR, exist_ok=True)
            _model.save(MODEL_DIR)
    return _model


def _length_score(candidate_answer, ideal_answer):
    """
    Penalize answers that are extremely short relative to the ideal answer,
    since high cosine similarity can be misleading for very short responses.
    """
    cand_len = len(candidate_answer.split())
    ideal_len = len(ideal_answer.split())

    if ideal_len == 0:
        return 1.0

    ratio = cand_len / ideal_len
    if ratio >= 0.5:
        return 1.0
    elif ratio <= 0.05:
        return 0.2
    else:
        # linear scale between 0.2 and 1.0
        return 0.2 + (ratio - 0.05) * (0.8 / 0.45)


def evaluate_answer(candidate_answer, ideal_answer, keywords=None):
    """
    Args:
        candidate_answer: user's spoken/transcribed answer (str)
        ideal_answer: reference/expected answer (str)
        keywords: optional list of key terms expected in a good answer

    Returns:
        {
            "similarity_score": float (0-100),
            "keyword_score": float (0-100) or None,
            "final_score": float (0-100)
        }
    """
    model = get_model()

    if not candidate_answer.strip():
        return {
            "similarity_score": 0.0,
            "keyword_score": 0.0 if keywords else None,
            "final_score": 0.0
        }

    embeddings = model.encode([candidate_answer, ideal_answer])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    sim = max(0.0, min(1.0, sim))  # clamp

    length_factor = _length_score(candidate_answer, ideal_answer)
    similarity_score = sim * length_factor * 100

    keyword_score = None
    if keywords:
        candidate_lower = candidate_answer.lower()
        matched = sum(1 for kw in keywords if kw.lower() in candidate_lower)
        keyword_score = (matched / len(keywords)) * 100 if keywords else 0.0

    if keyword_score is not None:
        final_score = 0.7 * similarity_score + 0.3 * keyword_score
    else:
        final_score = similarity_score

    return {
        "similarity_score": round(similarity_score, 2),
        "keyword_score": round(keyword_score, 2) if keyword_score is not None else None,
        "final_score": round(final_score, 2)
    }


def evaluate_answers_batch(qa_pairs):
    """
    qa_pairs: list of dicts {candidate_answer, ideal_answer, keywords (optional)}
    Returns list of evaluation results + average final score.
    """
    results = []
    for pair in qa_pairs:
        res = evaluate_answer(
            pair["candidate_answer"],
            pair["ideal_answer"],
            pair.get("keywords")
        )
        results.append(res)

    avg_score = sum(r["final_score"] for r in results) / len(results) if results else 0
    return {
        "average_score": round(avg_score, 2),
        "results": results
    }