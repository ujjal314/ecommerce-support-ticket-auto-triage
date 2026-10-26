
"""Evaluate the baseline on tagged language-variation subsets."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data.label_mapping import map_intent_to_route

RAW_PATH = Path(
    "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/baseline_model.joblib")

TAGS_TO_EVALUATE = {
    "Q": "Colloquial language",
    "Z": "Spelling errors / typos",
    "E": "Abbreviations",
    "W": "Offensive / frustrated language",
}


def clean_text(text):
    """Match the text cleaning used in preprocessing."""
    import re

    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def main():
    for path in (RAW_PATH, TEST_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    raw = pd.read_csv(RAW_PATH)
    test = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    # Reproduce the preprocessing order so raw tags can be
    # associated with the exact cleaned test examples.
    raw = raw.dropna(subset=["instruction", "intent"]).copy()
    raw["text"] = raw["instruction"].map(clean_text)
    raw["intent"] = raw["intent"].astype(str).str.strip().str.lower()
    raw = raw[raw["text"].str.len() > 0]
    raw = raw.drop_duplicates(subset=["text"]).reset_index(drop=True)
    raw["route_label"] = raw["intent"].map(map_intent_to_route)
    raw = raw[raw["route_label"].notna()].copy()

    tagged_test = test.merge(
        raw[["text", "route_label", "tags"]],
        on=["text", "route_label"],
        how="left",
        validate="one_to_one",
    )

    if tagged_test["tags"].isna().any():
        missing = int(tagged_test["tags"].isna().sum())
        raise ValueError(
            f"Could not match tags for {missing} test rows. "
            "Check preprocessing and text matching before interpreting results."
        )

    predictions = model.predict(tagged_test["text"])
    tagged_test["prediction"] = predictions

    def report(name, subset):
        if subset.empty:
            print(f"\n{name}: no examples found")
            return

        y_true = subset["route_label"]
        y_pred = subset["prediction"]

        # Average across the classes represented in this subset.
        labels = sorted(y_true.unique())

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        print(f"\n{name}")
        print(f"Examples: {len(subset):,}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

    report("Full held-out test set", tagged_test)

    tags = tagged_test["tags"].fillna("").astype(str)

    for tag, description in TAGS_TO_EVALUATE.items():
        mask = tags.str.contains(tag, regex=False)
        report(f"{tag} — {description}", tagged_test[mask])

    print(
        "\nNote: tagged subsets overlap. Their macro F1 scores "
        "use only the classes represented in each subset."
    )


if __name__ == "__main__":
    main()
