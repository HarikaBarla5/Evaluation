"""
Answer evaluation router
Endpoints:
  POST /evaluation/answer       - single Q&A -> similarity/keyword/final score
  POST /evaluation/answers-batch - multiple Q&A pairs -> per-question + average score
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.bert_evaluation import evaluate_answer, evaluate_answers_batch

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class AnswerRequest(BaseModel):
    candidate_answer: str
    ideal_answer: str
    keywords: Optional[List[str]] = None


class BatchAnswerRequest(BaseModel):
    qa_pairs: List[AnswerRequest]


@router.post("/answer")
async def evaluate_single(payload: AnswerRequest):
    try:
        result = evaluate_answer(
            payload.candidate_answer,
            payload.ideal_answer,
            payload.keywords
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/answers-batch")
async def evaluate_batch(payload: BatchAnswerRequest):
    try:
        qa_pairs = [p.dict() for p in payload.qa_pairs]
        result = evaluate_answers_batch(qa_pairs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")