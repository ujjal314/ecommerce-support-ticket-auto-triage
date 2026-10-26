
"""Compare both models on held-out language-variation subsets."""

from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data.label_mapping import map_intent_to_route

RAW_PATH = Path(
    "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
VAL_PATH = Path("data/processed/validation.csv")
BASELINE_PATH = Path("artifacts/baseline_model.joblib")
ROBUST_PATH = Path("artifacts/robust_baseline_model.joblib")

TAGS = {
    "ALL": "All validation examples",
    "Q": "Colloquial language",
    "Z": "Spelling errors / typos",
    "E": "Abbreviations",
    "W": "Offensive / frustrated language",
}


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def main():
    for path in (RAW_PATH, VAL_PATH, BASELINE_PATH, ROBUST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    raw = pd.read_csv(RAW_PATH)
    validation = pd.read_csv(VAL_PATH)

    raw = raw.dropna(subset=["instruction", "intent"]).copy()
    raw["text"] = raw["instruction"].map(clean_text)
    raw["intent"] = raw["intent"].astype(str).str.strip().str.lower()
    raw = raw[raw["text"].str.len() > 0]
    raw = raw.drop_duplicates(subset=["text"]).reset_index(drop=True)
    raw["route_label"] = raw["intent"].map(map_intent_to_route)
    raw = raw[raw["route_label"].notna()].copy()

    validation = validation.merge(
        raw[["text", "route_label", "tags"]],
        on=["text", "route_label"],
        how="left",
        validate="one_to_one",
    )

    if validation["tags"].isna().any():
        raise ValueError(
            "Some validation examples could not be matched to original tags."
        )

    baseline = joblib.load(BASELINE_PATH)
    robust = joblib.load(ROBUST_PATH)

    X = validation["text"]
    y = validation["route_label"]

    baseline_pred = baseline.predict(X)
    robust_pred = robust.predict(X)

    validation["baseline_prediction"] = baseline_pred
    validation["robust_prediction"] = robust_pred

    def evaluate(name, subset):
        if subset.empty:
            print(f"\n{name}: no examples")
            return

        y_true = subset["route_label"]
        labels = sorted(y_true.unique())

        print(f"\n{name} (n={len(subset):,})")

        for model_name, prediction_column in [
            ("Word-only", "baseline_prediction"),
            ("Word + character", "robust_prediction"),
        ]:
            pred = subset[prediction_column]
            accuracy = accuracy_score(y_true, pred)
            macro_f1 = f1_score(
                y_true,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
            print(
                f"  {model_name}: accuracy={accuracy:.4f}, "
                f"macro F1={macro_f1:.4f}"
            )

    tags = validation["tags"].fillna("").astype(str)

    for tag, description in TAGS.items():
        if tag == "ALL":
            subset = validation
        else:
            subset = validation[tags.str.contains(tag, regex=False)]

        evaluate(f"{tag} — {description}", subset)


if __name__ == "__main__":
    main()
