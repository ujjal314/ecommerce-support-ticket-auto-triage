
"""Compare word-only TF-IDF with combined word and character TF-IDF."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")
ARTIFACT_DIR = Path("artifacts")

BASELINE_PATH = ARTIFACT_DIR / "baseline_model.joblib"
ROBUST_MODEL_PATH = ARTIFACT_DIR / "robust_baseline_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "robust_baseline_validation_metrics.json"


def load_data(path):
    df = pd.read_csv(path)
    return df.dropna(subset=["text", "route_label"])


def build_model():
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=100000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def score_model(model, X, y):
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "predictions": predictions,
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_data(TRAIN_PATH)
    val = load_data(VAL_PATH)

    X_train, y_train = train["text"], train["route_label"]
    X_val, y_val = val["text"], val["route_label"]

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")

    # Evaluate the existing word-only model on the same validation set.
    baseline = joblib.load(BASELINE_PATH)
    baseline_scores = score_model(baseline, X_val, y_val)

    print("\nExisting word-only baseline:")
    print(f"Accuracy: {baseline_scores['accuracy']:.4f}")
    print(f"Macro F1: {baseline_scores['macro_f1']:.4f}")

    # Train the new model using training data only.
    robust_model = build_model()
    print("\nTraining word + character TF-IDF model...")
    robust_model.fit(X_train, y_train)

    robust_scores = score_model(robust_model, X_val, y_val)

    print("\nCombined word + character model:")
    print(f"Accuracy: {robust_scores['accuracy']:.4f}")
    print(f"Macro F1: {robust_scores['macro_f1']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_val,
        robust_scores["predictions"],
        zero_division=0,
    ))

    results = {
        "evaluation_split": "validation",
        "baseline_word_only": {
            "accuracy": baseline_scores["accuracy"],
            "macro_f1": baseline_scores["macro_f1"],
        },
        "word_and_character_tfidf": {
            "accuracy": robust_scores["accuracy"],
            "macro_f1": robust_scores["macro_f1"],
        },
    }

    joblib.dump(robust_model, ROBUST_MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model: {ROBUST_MODEL_PATH.resolve()}")
    print(f"Saved metrics: {METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
