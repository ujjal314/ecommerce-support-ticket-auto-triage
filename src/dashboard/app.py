
"""Streamlit dashboard for e-commerce support ticket triage."""

import streamlit as st
import csv
from datetime import datetime, timezone
from pathlib import Path
import uuid

from src.services.triage import triage_ticket

FEEDBACK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "feedback"
    / "reviewer_feedback.csv"
)

ROUTING_LABELS = [
    "refunds_cancellations",
    "damaged_missing_items",
    "shipping_logistics",
    "account_technical",
    "payments_billing",
    "product_catalog_availability",
    "order_management",
    "store_information",
    "customer_feedback",
    "sales_promotions",
    "human_triage_unknown",
]


st.set_page_config(
    page_title="Support Ticket Auto-Triage",
    page_icon="🎫",
    layout="wide",
)

st.title("🎫 E-Commerce Support Ticket Auto-Triage")
st.caption(
    "Classify incoming support tickets and identify cases "
    "that may need human review."
)

EXAMPLES = {
    "Refund status": "Where is my refund? I returned my order last week.",
    "Damaged or missing items": (
        "My package arrived damaged and some items are missing."
    ),
    "Account access": "I cannot reset my password.",
    "Payment problem": "I was charged twice for the same order.",
    "Delivery tracking": "My package has not arrived. Where is it?",
    "Product availability": "Is this product available online?",
    "Unknown request": "asdf zzz unknown request",
}

st.subheader("1. Enter a support ticket")

selected_example = st.selectbox(
    "Try an example ticket",
    ["Choose an example..."] + list(EXAMPLES.keys()),
)

if st.button("Load selected example"):
    if selected_example != "Choose an example...":
        st.session_state["ticket_text"] = EXAMPLES[selected_example]

ticket_text = st.text_area(
    "Customer message",
    key="ticket_text",
    height=140,
    max_chars=10000,
    placeholder="For example: I need help getting a refund for my order.",
)

st.subheader("2. Configure the review threshold")

confidence_threshold = st.slider(
    "Minimum confidence score for automatic routing",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
    help=(
        "Predictions below this threshold are assigned to the "
        "human-review queue. This threshold is configurable and "
        "has not yet been calibrated on real-world tickets."
    ),
)

st.caption(
    "The confidence score is the model's estimated class probability, "
    "not a calibrated guarantee that its prediction is correct."
)

if st.button("Classify ticket", type="primary"):
    if not ticket_text or not ticket_text.strip():
        st.warning("Please enter a customer message first.")
    else:
        try:
            with st.spinner("Classifying ticket..."):
                result = triage_ticket(
                    text=ticket_text,
                    confidence_threshold=confidence_threshold,
                )

            st.session_state["triage_result"] = result

        except (ValueError, FileNotFoundError) as exc:
            st.error(f"Unable to classify this ticket: {exc}")
        except Exception:
            st.error(
                "An unexpected error occurred while classifying "
                "the ticket. Check the terminal for details."
            )

result = st.session_state.get("triage_result")

if result:
    st.divider()
    st.subheader("3. Triage result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted category",
            result["predicted_label"].replace("_", " ").title(),
        )

    with col2:
        st.metric(
            "Model confidence",
            f'{result["confidence"]:.2%}',
        )

    with col3:
        st.metric(
            "Assigned queue",
            result["assigned_queue"].replace("_", " ").title(),
        )

    if result["needs_human_review"]:
        st.warning(
            "Human review required: the confidence score is below "
            "the configured threshold."
        )
    else:
        st.success(
            "The ticket meets the configured threshold for "
            "automatic routing. This does not guarantee correctness."
        )

    with st.expander("View prediction details"):
        st.write("**Ticket text:**", result["ticket_text"])
        st.write("**Predicted label:**", result["predicted_label"])
        st.write("**Assigned queue:**", result["assigned_queue"])
        st.write("**Confidence:**", result["confidence"])
        st.write(
            "**Review threshold:**",
            result["confidence_threshold"],
        )
        st.write(
            "**Human review required:**",
            result["needs_human_review"],
        )

st.divider()
st.caption(
    "Prototype for decision support. Validate model performance "
    "and review policies before using it for live customer tickets."
)


if result:
    st.divider()
    st.subheader("4. Reviewer feedback")

    st.write(
        "If the prediction or routing decision is incorrect, "
        "select the queue a reviewer believes is appropriate."
    )

    corrected_queue = st.selectbox(
        "Correct queue",
        ROUTING_LABELS,
        key="corrected_queue",
        format_func=lambda label: label.replace("_", " ").title(),
    )

    reviewer_notes = st.text_input(
        "Reviewer notes (optional)",
        key="reviewer_notes",
        max_chars=500,
    )

    if st.button("Save reviewer feedback"):
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

        feedback_exists = FEEDBACK_PATH.exists()

        with FEEDBACK_PATH.open(
            "a", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "feedback_id",
                    "timestamp_utc",
                    "ticket_text",
                    "predicted_label",
                    "assigned_queue",
                    "confidence",
                    "corrected_queue",
                    "reviewer_notes",
                ],
            )

            if not feedback_exists or FEEDBACK_PATH.stat().st_size == 0:
                writer.writeheader()

            writer.writerow({
                "feedback_id": str(uuid.uuid4()),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ticket_text": result["ticket_text"],
                "predicted_label": result["predicted_label"],
                "assigned_queue": result["assigned_queue"],
                "confidence": result["confidence"],
                "corrected_queue": corrected_queue,
                "reviewer_notes": reviewer_notes.strip(),
            })

        st.success("Reviewer feedback saved successfully.")
