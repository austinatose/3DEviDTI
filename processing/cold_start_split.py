#!/usr/bin/env python3
"""Cold-start splitting utility.

Goal:
  - Read existing train/val/test CSVs, collate into one dataset (drop duplicates).
  - Create a *drug cold-start* split where the TEST set contains *all interactions* for
    a randomly selected set of drugs comprising 10% of unique drugs.
  - Use all remaining interactions for TRAIN.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def cold_start_split_by_drug(
    df: pd.DataFrame,
    drug_col: str,
    test_drug_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (train_df, test_df, test_drugs).

    test_drugs are sampled uniformly from unique drugs.
    test_df contains *all* rows whose drug is in test_drugs.
    train_df contains all remaining interactions.
    """

    if not (0.0 < test_drug_frac < 1.0):
        raise ValueError("test_drug_frac must be in (0, 1)")
    if drug_col not in df.columns:
        raise KeyError(f"drug_col '{drug_col}' not found in dataframe columns")

    # Ensure consistent indexing
    df = df.reset_index(drop=True)

    # Sample drugs for test
    unique_drugs = pd.Series(df[drug_col].dropna().unique())
    if len(unique_drugs) == 0:
        raise ValueError("No drugs found (unique_drugs is empty). Check drug_col.")

    n_test_drugs = max(1, int(round(len(unique_drugs) * test_drug_frac)))
    test_drugs = unique_drugs.sample(n=n_test_drugs, random_state=seed).reset_index(drop=True)

    is_test = df[drug_col].isin(set(test_drugs.tolist()))
    test_df = df[is_test].reset_index(drop=True)
    remain_df = df[~is_test].reset_index(drop=True)

    # All remaining interactions go to train
    if len(remain_df) == 0:
        raise ValueError(
            "After selecting test drugs, no interactions remain for train. "
            "Reduce test_drug_frac or check the dataset."
        )

    train_df = remain_df.reset_index(drop=True)

    return train_df, test_df, test_drugs


def main() -> None:
    p = argparse.ArgumentParser(description="Create a drug cold-start split from existing CSV splits")
    p.add_argument("--train", required=True, type=Path, help="Path to current train CSV")
    p.add_argument("--valid", required=True, type=Path, help="Path to current valid CSV")
    p.add_argument("--test", required=True, type=Path, help="Path to current test CSV")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory")

    p.add_argument(
        "--drug_col",
        default=None,
        help="Drug ID column name. If omitted, will try common defaults.",
    )
    p.add_argument(
        "--prot_col",
        default=None,
        help="Protein ID column name (not required for splitting, only for validation/logging).",
    )

    p.add_argument("--test_drug_frac", type=float, default=0.10, help="Fraction of unique drugs in test")
    p.add_argument("--seed", type=int, default=2, help="Random seed")
    p.add_argument(
        "--dedupe_subset",
        default=None,
        help=(
            "Optional comma-separated columns for duplicate removal (e.g. 'drugbank_id,uniprot_id,Label'). "
            "If omitted, full-row duplicates are removed."
        ),
    )

    args = p.parse_args()

    # Read and collate
    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)
    test_df = pd.read_csv(args.test)

    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # Infer columns if not provided
    inferred_drug_col = args.drug_col or _pick_first_existing_col(
        df,
        ["drugbank_id", "drug_id", "Drug", "SMILES", "0"],
    )
    if inferred_drug_col is None:
        raise SystemExit(f"Could not infer drug_col. Available columns: {list(df.columns)}")

    # De-duplicate
    if args.dedupe_subset:
        subset_cols = [c.strip() for c in args.dedupe_subset.split(",") if c.strip()]
        missing = [c for c in subset_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"dedupe_subset columns not found: {missing}. Available: {list(df.columns)}")
        df = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
    else:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

    train_out, test_out, test_drugs = cold_start_split_by_drug(
        df=df,
        drug_col=inferred_drug_col,
        test_drug_frac=args.test_drug_frac,
        seed=args.seed,
    )

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / "train.csv"
    test_path = args.out_dir / "test.csv"
    drugs_path = args.out_dir / "test_drugs.txt"

    def count_pos_neg(df, label_col="interaction"):
        return df[label_col].value_counts().sort_index()

    print("Train:")
    print(count_pos_neg(train_out))

    print("\nTest:")
    print(count_pos_neg(test_out))

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    drugs_path.write_text("\n".join(map(str, test_drugs.tolist())) + "\n")

    # Print a short summary
    print("=== Cold-start (drug) split created ===")
    print(f"Input rows (after dedupe): {len(df):,}")
    print(f"Unique drugs: {df[inferred_drug_col].nunique():,}")
    print(f"Test drugs: {len(test_drugs):,} ({args.test_drug_frac:.2%} of unique drugs)")
    print(f"Rows -> train: {len(train_out):,} | test: {len(test_out):,}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {drugs_path}")


if __name__ == "__main__":
    main()
