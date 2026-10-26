
"""Save and inspect errors made by the baseline on the test set."""

from pathlib import Path
import re

import joblib
import pandas as pd

from src.data.label_mapping import map_intent_to_route

RAW_PATH = Path(
    "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("artifacts/baseline_model.joblib")
OUTPUT_PATH = Path("artifacts/test_misclassified.csv")


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def main():
    raw = pd.read_csv(RAW_PATH)
    test = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    # Reproduce the preprocessing steps used for the dataset.
    raw = raw.dropna(subset=["instruction", "intent"]).copy()
    raw["text"] = raw["instruction"].map(clean_text)
    raw["intent"] = raw["intent"].astype(str).str.strip().str.lower()
    raw = raw[raw["text"].str.len() > 0]
    raw = raw.drop_duplicates(subset=["text"]).reset_index(drop=True)
    raw["route_label"] = raw["intent"].map(map_intent_to_route)
    raw = raw[raw["route_label"].notna()].copy()

    test = test.merge(
        raw[["text", "route_label", "tags"]],
        on=["text", "route_label"],
        how="left",
        validate="one_to_one",
    )

    if test["tags"].isna().any():
        raise ValueError(
            "Some test examples could not be matched to their original tags."
        )

    test["predicted_label"] = model.predict(test["text"])
    errors = test[test["route_label"] != test["predicted_label"]].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_PATH, index=False)

    print(f"Total test examples: {len(test):,}")
    print(f"Misclassified examples: {len(errors):,}")
    print(f"Error rate: {len(errors) / len(test):.2%}")
    print(f"Saved errors to: {OUTPUT_PATH.resolve()}")

    if errors.empty:
        print("No misclassified examples found.")
        return

    print("\nMost common true-label -> predicted-label errors:")
    pairs = (
        errors.groupby(["route_label", "predicted_label"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print(pairs.to_string())

    typo_errors = errors[
        errors["tags"].fillna("").str.contains("Z", regex=False)
    ]

    print(f"\nMisclassified examples tagged with Z: {len(typo_errors)}")
    print("\nSample errors:")
    columns = ["text", "route_label", "predicted_label", "tags"]
    print(errors[columns].head(15).to_string(index=False))

    if not typo_errors.empty:
        print("\nSample typo-related errors:")
        print(typo_errors[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
