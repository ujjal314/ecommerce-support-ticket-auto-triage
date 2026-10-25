
"""Automated tests for the FastAPI ticket-triage endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_triage_success(monkeypatch):
    def fake_triage_ticket(text, confidence_threshold):
        return {
            "ticket_text": text.strip(),
            "predicted_label": "refunds_cancellations",
            "confidence": 0.95,
            "assigned_queue": "refunds_cancellations",
            "needs_human_review": False,
            "confidence_threshold": confidence_threshold,
        }

    monkeypatch.setattr(
        "src.api.main.triage_ticket",
        fake_triage_ticket,
    )

    response = client.post(
        "/triage",
        json={"text": "Where is my refund?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "refunds_cancellations"
    assert body["assigned_queue"] == "refunds_cancellations"
    assert body["needs_human_review"] is False


def test_low_confidence_ticket_goes_to_review(monkeypatch):
    def fake_triage_ticket(text, confidence_threshold):
        return {
            "ticket_text": text,
            "predicted_label": "order_management",
            "confidence": 0.30,
            "assigned_queue": "human_triage_unknown",
            "needs_human_review": True,
            "confidence_threshold": confidence_threshold,
        }

    monkeypatch.setattr(
        "src.api.main.triage_ticket",
        fake_triage_ticket,
    )

    response = client.post(
        "/triage",
        json={"text": "I was charged twice for the same order"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assigned_queue"] == "human_triage_unknown"
    assert body["needs_human_review"] is True


def test_empty_text_is_rejected():
    response = client.post(
        "/triage",
        json={"text": ""},
    )

    assert response.status_code == 422


def test_invalid_confidence_threshold_is_rejected():
    response = client.post(
        "/triage",
        json={
            "text": "Where is my refund?",
            "confidence_threshold": 1.5,
        },
    )

    assert response.status_code == 422


def test_missing_text_is_rejected():
    response = client.post("/triage", json={})

    assert response.status_code == 422


def test_model_unavailable_returns_503(monkeypatch):
    def fake_triage_ticket(text, confidence_threshold):
        raise FileNotFoundError("Model file missing")

    monkeypatch.setattr(
        "src.api.main.triage_ticket",
        fake_triage_ticket,
    )

    response = client.post(
        "/triage",
        json={"text": "Where is my refund?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The trained model is unavailable."
    )
