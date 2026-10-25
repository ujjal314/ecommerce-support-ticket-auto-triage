
from pathlib import Path
import pandas as pd

DATA_PATH = Path(
    "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head().to_string())

for col in ["category", "intent"]:
    if col in df.columns:
        print(f"\nDistribution of {col}:")
        print(df[col].value_counts().to_string())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())
