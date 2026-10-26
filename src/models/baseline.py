
"""Train a TF-IDF + Logistic Regression ticket-routing baseline."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "baseline_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "baseline_metrics.json"


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}. "
            "Run python -m src.data.preprocess first."
        )

    df = pd.read_csv(path)

    required_columns = {"text", "route_label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{path} must contain {required_columns}"
        )

    return df.dropna(subset=["text", "route_label"])


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VALIDATION_PATH)

    # TF-IDF learns vocabulary and IDF weights from training data only.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print("\nTraining baseline model...")

    model.fit(train_df["text"], train_df["route_label"])

    predictions = model.predict(val_df["text"])
    labels = sorted(train_df["route_label"].unique())

    accuracy = accuracy_score(val_df["route_label"], predictions)
    macro_f1 = f1_score(
        val_df["route_label"],
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        val_df["route_label"],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        val_df["route_label"],
        predictions,
        labels=labels,
    )

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "evaluation_split": "validation",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_label_order": labels,
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            val_df["route_label"],
            predictions,
            labels=labels,
            zero_division=0,
        )
    )
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))
    print(f"\nSaved model: {MODEL_PATH.resolve()}")
    print(f"Saved metrics: {METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
