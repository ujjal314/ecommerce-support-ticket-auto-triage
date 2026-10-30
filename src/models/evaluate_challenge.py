
"""Evaluate routing behavior on independently authored challenge tickets."""

import csv
from collections import defaultdict
from pathlib import Path

from src.services.triage import triage_ticket

CHALLENGE_PATH = Path("data/challenge/challenge_tickets.csv")
CONFIDENCE_THRESHOLD = 0.70


def main():
    if not CHALLENGE_PATH.exists():
        raise FileNotFoundError(
            f"Challenge dataset not found: {CHALLENGE_PATH.resolve()}"
        )

    with CHALLENGE_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        tickets = list(csv.DictReader(file))

    if not tickets:
        raise ValueError("The challenge dataset is empty.")

    required = {"ticket_text", "expected_route", "challenge_type"}
    if not required.issubset(tickets[0]):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    supported_total = 0
    supported_correct = 0
    review_total = 0
    review_correct = 0
    category_results = defaultdict(lambda: [0, 0])

    print(f"Challenge tickets: {len(tickets)}")
    print(f"Human-review threshold: {CONFIDENCE_THRESHOLD:.2f}\n")

    for index, ticket in enumerate(tickets, start=1):
        expected = ticket["expected_route"].strip()
        result = triage_ticket(
            ticket["ticket_text"],
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        if expected == "human_review":
            review_total += 1
            correct = result["needs_human_review"]
            review_correct += int(correct)
        else:
            supported_total += 1
            correct = (
                result["assigned_queue"] == expected
                and not result["needs_human_review"]
            )
            supported_correct += int(correct)

            category_results[expected][0] += int(correct)
            category_results[expected][1] += 1

        status = "PASS" if correct else "CHECK"
        print(
            f"{index:02d}. [{status}] "
            f"expected={expected} | "
            f"predicted={result['predicted_label']} | "
            f"queue={result['assigned_queue']} | "
            f"confidence={result['confidence']:.3f}"
        )
        print(f"    Ticket: {ticket['ticket_text']}")

    print("\n--- Summary ---")

    if supported_total:
        print(
            "Supported-ticket exact routing accuracy: "
            f"{supported_correct}/{supported_total} "
            f"({supported_correct / supported_total:.1%})"
        )

    if review_total:
        print(
            "Human-review detection rate on review-required cases: "
            f"{review_correct}/{review_total} "
            f"({review_correct / review_total:.1%})"
        )

    print("\nSupported categories:")
    for category, (correct, total) in sorted(category_results.items()):
        print(f"  {category}: {correct}/{total} ({correct / total:.1%})")

    print(
        "\nInterpretation: this is a small diagnostic challenge set, "
        "not a statistically representative production benchmark."
    )


if __name__ == "__main__":
    main()
