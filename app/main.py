"""FastAPI application -- language feedback endpoint."""

import logging
 
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
 
from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

load_dotenv()

logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback using Claude.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        return await get_feedback(request)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
 