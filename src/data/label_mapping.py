
"""Map Bitext support intents to operational ticket-routing queues."""

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

# The dataset does not provide a dedicated fraud/security intent.
# Keep this as a documented coverage gap rather than inventing labels.
UNSUPPORTED_PRD_LABELS = ["fraud_security"]

INTENT_TO_ROUTE = {
    # Refunds, returns, exchanges, cancellations
    "request_refund": "refunds_cancellations",
    "refund_status": "refunds_cancellations",
    "refund_policy": "refunds_cancellations",
    "return_policy": "refunds_cancellations",
    "return_product": "refunds_cancellations",
    "return_product_in_store": "refunds_cancellations",
    "return_product_online": "refunds_cancellations",
    "exchange_product": "refunds_cancellations",
    "exchange_product_in_store": "refunds_cancellations",
    "cancel_order": "refunds_cancellations",

    # Damaged, missing, incorrect, or unclear product problems
    "damaged_delivery": "damaged_missing_items",
    "missing_item": "damaged_missing_items",
    "wrong_item": "damaged_missing_items",

    # Shipping and delivery
    "delivery_issue": "shipping_logistics",
    "delivery_time": "shipping_logistics",
    "track_delivery": "shipping_logistics",
    "track_order": "shipping_logistics",
    "shipping_costs": "shipping_logistics",

    # Account and technical support
    "change_account": "account_technical",
    "close_account": "account_technical",
    "open_account": "account_technical",
    "recover_password": "account_technical",
    "technical_issue": "account_technical",
    "use_app": "account_technical",
    "request_right_to_rectification": "account_technical",

    # Payments and invoices
    "payment_issue": "payments_billing",
    "payment_methods": "payments_billing",
    "pay": "payments_billing",
    "request_invoice": "payments_billing",

    # Product information and availability
    "product_information": "product_catalog_availability",
    "availability": "product_catalog_availability",
    "availability_online": "product_catalog_availability",
    "availability_in_store": "product_catalog_availability",

    # Cart and order modifications
    "change_order": "order_management",
    "add_product": "order_management",
    "remove_product": "order_management",
    "order_history": "order_management",

    # Physical store information
    "store_location": "store_information",
    "store_opening_hours": "store_information",

    # Feedback and suggestions
    "submit_feedback": "customer_feedback",
    "submit_product_feedback": "customer_feedback",
    "submit_product_idea": "customer_feedback",

    # Sales and promotions
    "sales_period": "sales_promotions",

    # General requests that may need human review
    "human_agent": "human_triage_unknown",
    "customer_service": "human_triage_unknown",
    "product_issue": "human_triage_unknown",
}


def map_intent_to_route(intent: str) -> str | None:
    """Return a supported operational queue or None if unmapped."""
    normalized = str(intent).strip().lower()
    return INTENT_TO_ROUTE.get(normalized)
