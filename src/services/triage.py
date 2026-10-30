
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


import re

def requires_rule_based_review(text: str) -> bool:
    """Flag sensitive or high-risk requests for human review."""

    normalized = text.lower()

    if (
        "transaction" in normalized
        and any(
            phrase in normalized
            for phrase in (
                "don't recognise",
                "dont recognise",
                "do not recognise",
                "don't recognize",
                "dont recognize",
                "do not recognize",
                "don't recognize",
                "unrecognized transaction",
                "unrecognised transaction",
            )
        )
    ):
        return True

    security_patterns = [
        r"\b(?:hacked|compromised|phishing|fraud|scam)\b",
        r"\bunauthori[sz]ed\b",
        r"\b(?:someone|somebody) (?:may have )?accessed my account\b",
        r"\b(?:don't|do not) recognis[sz]e (?:a )?transaction\b",
        r"\bunrecognis[sz]ed transaction\b",
        r"\bsuspicious transaction\b",
        r"\b(?:card|payment) details (?:were )?changed\b",
        r"\blink asking for my card details\b",
        r"\bpossible identity theft\b",
    ]

    for pattern in security_patterns:
        if re.search(pattern, normalized):
            return True

    # A shipped order with an address-change request needs review.
    shipped_address_change = (
        re.search(r"\bshipped\b", normalized)
        and re.search(r"\b(?:change|update|correct)\b", normalized)
        and re.search(r"\baddress\b", normalized)
    )
    if shipped_address_change:
        return True

    # A duplicate charge combined with cancellation is multi-intent.
    duplicate_charge = re.search(
        r"\b(?:charged twice|charged two times|double charged)\b",
        normalized,
    )
    cancellation = re.search(
        r"\b(?:cancel|cancellation)\b",
        normalized,
    )


    if duplicate_charge and cancellation:
        return True

    # A missing refund combined with return-status uncertainty
    # should be reviewed by a human.
    refund_return_status = (
        re.search(r"\brefund\b", normalized)
        and re.search(r"\bmissing\b", normalized)
        and re.search(r"\breturn\b", normalized)
        and re.search(r"\breceived\b", normalized)
    )

    if refund_return_status:
        return True

    return False



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

    needs_human_review = (
        confidence < confidence_threshold
        or predicted_label == HUMAN_REVIEW_LABEL
        or requires_rule_based_review(text.strip())
    )

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
