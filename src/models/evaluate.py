
"""Evaluate the saved baseline on the held-out test set."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

MODEL_PATH = Path("artifacts/baseline_model.joblib")
TEST_PATH = Path("data/processed/test.csv")
OUTPUT_PATH = Path("artifacts/test_metrics.json")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Baseline model not found. Run "
            "python -m src.models.baseline first."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "Test data not found. Run "
            "python -m src.data.preprocess first."
        )

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH).dropna(
        subset=["text", "route_label"]
    )

    y_true = test_df["route_label"]
    y_pred = model.predict(test_df["text"])
    labels = sorted(y_true.unique())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    results = {
        "model": "TF-IDF + Logistic Regression",
        "evaluation_split": "test",
        "test_rows": len(test_df),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_label_order": labels,
    }

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"Test rows: {len(test_df):,}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true, y_pred,
            labels=labels,
            zero_division=0,
        )
    )
    print("\nConfusion matrix:")
    print(pd.DataFrame(
        matrix, index=labels, columns=labels
    ))
    print(f"\nResults saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
