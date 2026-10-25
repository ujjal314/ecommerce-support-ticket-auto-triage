
"""FastAPI application for the e-commerce ticket triage system."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.services.triage import triage_ticket

app = FastAPI(
    title="E-Commerce Support Ticket Auto-Triage API",
    description=(
        "Classifies support tickets and recommends an operational queue."
    ),
    version="0.1.0",
)


class TriageRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Customer support ticket text",
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for automatic queue assignment",
    )


class TriageResponse(BaseModel):
    ticket_text: str
    predicted_label: str
    confidence: float
    assigned_queue: str
    needs_human_review: bool
    confidence_threshold: float


@app.get("/health")
def health():
    """Check whether the API process is running."""
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    """Classify a ticket and return its recommended routing decision."""
    try:
        return triage_ticket(
            text=request.text,
            confidence_threshold=request.confidence_threshold,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The trained model is unavailable.",
        ) from exc
