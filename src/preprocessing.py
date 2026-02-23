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


def create_stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    stratify_col: str = "product_category",
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if stratify_col not in df.columns:
        raise ValueError(f"Dataframe must include '{stratify_col}' for stratified sampling.")
    if sample_size >= len(df):
        return df.copy()

    group_counts = df[stratify_col].value_counts()
    proportions = group_counts / group_counts.sum()

    targets = (proportions * sample_size).round().astype(int)
    targets = targets.clip(lower=1)

    while targets.sum() > sample_size:
        reducible = targets[targets > 1]
        if reducible.empty:
            break
        largest_group = reducible.idxmax()
        targets.loc[largest_group] -= 1

    while targets.sum() < sample_size:
        add_to_group = (proportions - (targets / sample_size)).idxmax()
        targets.loc[add_to_group] += 1

    sampled_groups = []
    for group_name, target in targets.items():
        group_df = df[df[stratify_col] == group_name]
        n = min(int(target), len(group_df))
        sampled_groups.append(group_df.sample(n=n, random_state=random_state))

    sampled_df = pd.concat(sampled_groups, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return sampled_df


def run_preprocessing(
    input_csv: Path,
    output_csv: Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict:
    raw_df = pd.read_csv(input_csv, low_memory=False)
    processed_df = preprocess_dataframe(raw_df)

    if sample_size is not None:
        processed_df = create_stratified_sample(
            processed_df,
            sample_size=sample_size,
            stratify_col="product_category",
            random_state=random_state,
        )

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
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size to reduce dataset volume (e.g., 10000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preprocessing(
        args.input_csv,
        args.output_csv,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print("Preprocessing complete")
    print(f"Rows before: {summary['rows_before']}")
    print(f"Rows after: {summary['rows_after']}")
    print(f"Average word count: {summary['avg_word_count']:.2f}")
    if args.sample_size is not None:
        print(f"Applied stratified sample size: {args.sample_size}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()