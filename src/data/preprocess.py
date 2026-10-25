
"""Clean Bitext data, map supported labels, and create reproducible splits."""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.label_mapping import (
    INTENT_TO_ROUTE,
    ROUTING_LABELS,
    map_intent_to_route,
)

RAW_PATH = Path(
    "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("artifacts")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def clean_text(text: str) -> str:
    """Remove simple HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH.resolve()}"
        )

    df = pd.read_csv(RAW_PATH)

    required = {"instruction", "intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["instruction", "intent"]).copy()
    df["text"] = df["instruction"].map(clean_text)
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()

    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Unsupported intents remain unmapped; they are not training labels.
    df["route_label"] = df["intent"].map(map_intent_to_route)

    unsupported_counts = (
        df.loc[df["route_label"].isna(), "intent"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Train only on intents with a documented routing destination.
    df = df[df["route_label"].notna()].copy()

    if df.empty:
        raise ValueError("No rows have supported routing labels.")

    # DEMONSTRATION ONLY: these are synthetic values, not real customer data.
    rng = np.random.default_rng(RANDOM_STATE)
    df["order_value"] = np.round(
        np.clip(
            rng.lognormal(mean=4.0, sigma=0.8, size=len(df)),
            1,
            5000,
        ),
        2,
    )
    df["customer_tier"] = rng.choice(
        [1, 2, 3],
        size=len(df),
        p=[0.60, 0.30, 0.10],
    )

    result = df[
        ["text", "intent", "route_label", "order_value", "customer_tier"]
    ].reset_index(drop=True)

    return result, unsupported_counts


def split_data(df: pd.DataFrame):
    """Create approximately 70% train, 15% validation, 15% test."""
    counts = df["route_label"].value_counts()

    if counts.min() < 3:
        raise ValueError(
            "Each routing class needs at least 3 examples for splitting."
        )

    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["route_label"],
    )

    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)

    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=train_val["route_label"],
    )

    return train, validation, test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, unsupported_counts = load_and_prepare()
    train, validation, test = split_data(df)

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, split in splits.items():
        path = OUTPUT_DIR / f"{name}.csv"
        split.to_csv(path, index=False)
        print(f"Saved {path}: {len(split):,} rows")

    report = {
        "random_state": RANDOM_STATE,
        "supported_rows": len(df),
        "split_sizes": {
            name: len(split) for name, split in splits.items()
        },
        "routing_label_counts": {
            str(k): int(v)
            for k, v in df["route_label"].value_counts().items()
        },
        "routing_labels": ROUTING_LABELS,
        "supported_intents": sorted(INTENT_TO_ROUTE),
        "unsupported_intent_counts": {
            str(k): int(v) for k, v in unsupported_counts.items()
        },
        "synthetic_metadata": {
            "order_value": True,
            "customer_tier": True,
            "warning": (
                "Randomly generated demonstration values; "
                "not genuine customer or order records."
            ),
        },
        "duplicate_texts_removed_before_splitting": True,
    }

    report_path = REPORT_DIR / "dataset_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSupported routing labels:")
    print(df["route_label"].value_counts().to_string())
    print("\nUnsupported intents excluded from training:")
    print(json.dumps(unsupported_counts, indent=2))
    print(f"\nReport saved to {report_path.resolve()}")


if __name__ == "__main__":
    main()
