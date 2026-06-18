"""
Score Engine
Combines:
  - Emotion score (CNN, 0-100)
  - Voice confidence + pace score (LSTM, 0-100 each)
  - Answer evaluation score (BERT, 0-100, per question -> averaged)

Produces a final weighted interview score + breakdown + textual feedback.
"""

# ---------------- WEIGHTS ----------------
# Adjust based on what matters most for your evaluation criteria.
WEIGHTS = {
    "emotion": 0.20,
    "voice_confidence": 0.15,
    "voice_pace": 0.10,
    "answer": 0.55,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


def _band(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Average"
    elif score >= 35:
        return "Below Average"
    else:
        return "Needs Improvement"


def _feedback(emotion_score, voice_confidence, voice_pace, answer_score):
    feedback = []

    if emotion_score < 50:
        feedback.append("Try to maintain a calmer, more composed facial expression during answers.")
    elif emotion_score >= 80:
        feedback.append("Great composure and positive expressions throughout.")

    if voice_confidence < 50:
        feedback.append("Your voice tone suggests low confidence — practice speaking with a steadier, firmer tone.")
    elif voice_confidence >= 80:
        feedback.append("Strong, confident vocal delivery.")

    if voice_pace < 40:
        feedback.append("Your speaking pace was slow — try to respond a bit more fluidly.")
    elif voice_pace > 85:
        feedback.append("You spoke quite fast — slowing down slightly could improve clarity.")
    else:
        feedback.append("Your speaking pace was well balanced.")

    if answer_score < 50:
        feedback.append("Answers need more depth and alignment with expected key points.")
    elif answer_score >= 80:
        feedback.append("Answers were relevant and well-aligned with expected responses.")

    return feedback


def compute_final_score(emotion_score, voice_confidence, voice_pace, answer_score):
    """
    All inputs expected on 0-100 scale.

    Returns:
        {
            "final_score": float,
            "band": str,
            "breakdown": {...},
            "feedback": [str, ...]
        }
    """
    weighted = (
        emotion_score * WEIGHTS["emotion"] +
        voice_confidence * WEIGHTS["voice_confidence"] +
        voice_pace * WEIGHTS["voice_pace"] +
        answer_score * WEIGHTS["answer"]
    )

    final_score = round(weighted, 2)

    return {
        "final_score": final_score,
        "band": _band(final_score),
        "breakdown": {
            "emotion_score": round(emotion_score, 2),
            "voice_confidence_score": round(voice_confidence, 2),
            "voice_pace_score": round(voice_pace, 2),
            "answer_score": round(answer_score, 2),
            "weights": WEIGHTS,
        },
        "feedback": _feedback(emotion_score, voice_confidence, voice_pace, answer_score)
    }


def compute_interview_report(emotion_results, voice_results, answer_results):
    """
    Aggregates results from full interview session.

    Args:
        emotion_results: output of predict_emotion_sequence() -> {"average_score": ...}
        voice_results: list of predict_voice_score() outputs (one per question) OR a single dict
        answer_results: output of evaluate_answers_batch() -> {"average_score": ..., "results": [...]}

    Returns full report dict including per-question breakdown.
    """
    emotion_score = emotion_results.get("average_score", 0)

    if isinstance(voice_results, list):
        voice_confidence = sum(v["confidence_score"] for v in voice_results) / len(voice_results)
        voice_pace = sum(v["pace_score"] for v in voice_results) / len(voice_results)
    else:
        voice_confidence = voice_results.get("confidence_score", 0)
        voice_pace = voice_results.get("pace_score", 0)

    answer_score = answer_results.get("average_score", 0)

    summary = compute_final_score(emotion_score, voice_confidence, voice_pace, answer_score)

    return {
        **summary,
        "emotion_details": emotion_results,
        "voice_details": voice_results,
        "answer_details": answer_results,
    }