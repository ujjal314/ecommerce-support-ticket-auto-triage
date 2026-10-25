
"""Business logic for routing e-commerce support tickets."""

from pathlib import Path

import joblib

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "robust_baseline_model.joblib"
)

HUMAN_REVIEW_LABEL = "human_triage_unknown"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

_model = None


def load_model():
    """Load the trained model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. "
                "Run python -m src.models.robust_baseline first."
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def triage_ticket(
    text: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Predict a ticket queue and flag low-confidence predictions."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket text must be a non-empty string.")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    model = load_model()

    # Logistic Regression provides class probability estimates.
    probabilities = model.predict_proba([text.strip()])[0]
    classes = model.classes_

    best_index = int(probabilities.argmax())
    predicted_label = str(classes[best_index])
    confidence = float(probabilities[best_index])

    needs_human_review = confidence < confidence_threshold

    # Keep the model prediction visible for auditability, while
    # routing low-confidence tickets to the human-review queue.
    assigned_queue = (
        HUMAN_REVIEW_LABEL if needs_human_review else predicted_label
    )

    return {
        "ticket_text": text.strip(),
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "assigned_queue": assigned_queue,
        "needs_human_review": needs_human_review,
        "confidence_threshold": confidence_threshold,
    }
