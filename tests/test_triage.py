
import numpy as np

from src.services import triage


class FakeModel:
    def __init__(self, classes, probabilities):
        self.classes_ = np.array(classes)
        self.probabilities = np.array(probabilities)

    def predict_proba(self, texts):
        return np.array([self.probabilities])


def test_low_confidence_ticket_requires_review(monkeypatch):
    model = FakeModel(
        ["account_technical", "refunds_cancellations"],
        [0.60, 0.40],
    )
    monkeypatch.setattr(triage, "load_model", lambda: model)

    result = triage.triage_ticket(
        "I need help",
        confidence_threshold=0.70,
    )

    assert result["needs_human_review"] is True
    assert result["assigned_queue"] == "human_triage_unknown"


def test_human_triage_prediction_always_requires_review(monkeypatch):
    model = FakeModel(
        ["human_triage_unknown", "account_technical"],
        [0.95, 0.05],
    )
    monkeypatch.setattr(triage, "load_model", lambda: model)

    result = triage.triage_ticket(
        "I want to speak to a support agent",
        confidence_threshold=0.70,
    )

    assert result["predicted_label"] == "human_triage_unknown"
    assert result["needs_human_review"] is True
    assert result["assigned_queue"] == "human_triage_unknown"


def test_confident_supported_prediction_can_auto_route(monkeypatch):
    model = FakeModel(
        ["refunds_cancellations", "account_technical"],
        [0.95, 0.05],
    )
    monkeypatch.setattr(triage, "load_model", lambda: model)

    result = triage.triage_ticket(
        "Where is my refund?",
        confidence_threshold=0.70,
    )

    assert result["needs_human_review"] is False
    assert result["assigned_queue"] == "refunds_cancellations"



def test_security_request_requires_review(monkeypatch):
    model = FakeModel(
        ["account_technical", "payments_billing"],
        [0.98, 0.02],
    )
    monkeypatch.setattr(triage, "load_model", lambda: model)

    result = triage.triage_ticket(
        "I don't recognise a transaction on my account.",
        confidence_threshold=0.70,
    )

    assert result["needs_human_review"] is True
    assert result["assigned_queue"] == "human_triage_unknown"


def test_shipped_order_address_change_requires_review(monkeypatch):
    model = FakeModel(
        ["shipping_logistics", "order_management"],
        [0.98, 0.02],
    )
    monkeypatch.setattr(triage, "load_model", lambda: model)

    result = triage.triage_ticket(
        "My order has shipped, but I need to change the delivery address.",
        confidence_threshold=0.70,
    )

    assert result["needs_human_review"] is True
    assert result["assigned_queue"] == "human_triage_unknown"


def test_duplicate_charge_and_cancellation_require_review(monkeypatch):
    model = FakeModel(
        ["refunds_cancellations", "payments_billing"],
        [0.98, 0.02],
    )
    monkeypatch.setattr(triage, "load_model", lambda: model)

    result = triage.triage_ticket(
        "I was charged twice and I also want to cancel the order.",
        confidence_threshold=0.70,
    )

    assert result["needs_human_review"] is True
    assert result["assigned_queue"] == "human_triage_unknown"



def test_missing_refund_and_return_status_requires_review():
    from src.services.triage import requires_rule_based_review

    text = (
        "My refund is missing, and I need to know "
        "whether the return was received."
    )

    assert requires_rule_based_review(text) is True
