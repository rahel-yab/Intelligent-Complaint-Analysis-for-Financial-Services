from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PRODUCT_MAPPING = {
    "Credit card or prepaid card": "Credit Card",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Checking or savings account": "Savings Account",
    "Money transfer, virtual currency, or money service": "Money Transfer",
}


def clean_complaint_text(text: str) -> str:
    text = re.sub(r"X+", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = " ".join(text.split())
    return text.lower().strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Product" not in df.columns:
        raise ValueError("Input data must include a 'Product' column.")
    if "Consumer complaint narrative" not in df.columns:
        raise ValueError("Input data must include a 'Consumer complaint narrative' column.")

    filtered = df[df["Product"].isin(PRODUCT_MAPPING.keys())].copy()
    filtered["product_category"] = filtered["Product"].map(PRODUCT_MAPPING)

    filtered = filtered.dropna(subset=["Consumer complaint narrative"]).copy()
    filtered["Consumer complaint narrative"] = filtered["Consumer complaint narrative"].astype(str)
    filtered = filtered[filtered["Consumer complaint narrative"].str.strip().ne("")].copy()

    filtered["word_count"] = filtered["Consumer complaint narrative"].str.split().str.len()
    filtered["cleaned_narrative"] = filtered["Consumer complaint narrative"].apply(clean_complaint_text)
    filtered = filtered[filtered["cleaned_narrative"].str.strip().ne("")].copy()

    return filtered


def summarize_preprocessing(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    return {
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "dropped_rows": int(len(df_before) - len(df_after)),
        "avg_word_count": float(df_after["word_count"].mean()) if len(df_after) else 0.0,
        "product_distribution": df_after["product_category"].value_counts().to_dict() if len(df_after) else {},
    }


def run_preprocessing(input_csv: Path, output_csv: Path) -> dict:
    raw_df = pd.read_csv(input_csv, low_memory=False)
    processed_df = preprocess_dataframe(raw_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)

    return summarize_preprocessing(raw_df, processed_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CFPB complaints for RAG.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/row/complaints.csv"),
        help="Path to raw complaints CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/filtered_complaints.csv"),
        help="Path to save filtered and cleaned complaints CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preprocessing(args.input_csv, args.output_csv)
    print("Preprocessing complete")
    print(f"Rows before: {summary['rows_before']}")
    print(f"Rows after: {summary['rows_after']}")
    print(f"Average word count: {summary['avg_word_count']:.2f}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()