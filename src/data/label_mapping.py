
"""Explicitly map supported Bitext intents to operational departments."""

ROUTING_LABELS = [
    "refunds_cancellations",
    "damaged_missing_items",
    "shipping_logistics",
    "account_technical",
    "human_triage_unknown",
]

# Reserved PRD destination. No labelled examples are available in Bitext.
UNSUPPORTED_PRD_LABELS = ["fraud_security"]

INTENT_TO_ROUTE = {
    # Refunds, returns, cancellations
    "request_refund": "refunds_cancellations",
    "refund_status": "refunds_cancellations",
    "refund_policy": "refunds_cancellations",
    "return_policy": "refunds_cancellations",
    "return_product": "refunds_cancellations",
    "return_product_in_store": "refunds_cancellations",
    "return_product_online": "refunds_cancellations",
    "cancel_order": "refunds_cancellations",

    # Damaged, missing, or incorrect items
    "damaged_delivery": "damaged_missing_items",
    "missing_item": "damaged_missing_items",
    "wrong_item": "damaged_missing_items",

    # Shipping and logistics
    "delivery_issue": "shipping_logistics",
    "delivery_time": "shipping_logistics",
    "track_delivery": "shipping_logistics",
    "shipping_costs": "shipping_logistics",
    "track_order": "shipping_logistics",

    # Account and technical issues
    "change_account": "account_technical",
    "close_account": "account_technical",
    "open_account": "account_technical",
    "order_history": "account_technical",
    "recover_password": "account_technical",
    "technical_issue": "account_technical",
    "use_app": "account_technical",

    # Explicit request for a human
    "human_agent": "human_triage_unknown",
}


def map_intent_to_route(intent: str) -> str | None:
    """Return None for intents without a justified routing mapping."""
    normalized = str(intent).strip().lower()
    return INTENT_TO_ROUTE.get(normalized)
