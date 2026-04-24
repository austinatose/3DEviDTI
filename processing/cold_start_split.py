#!/usr/bin/env python3
"""Cold-start splitting utility.

Goal:
  - Read existing train/val/test CSVs, collate into one dataset (drop duplicates).
    - Create a cold-start split by drug or protein, where TEST and VAL each contain
        all interactions for disjoint sets of entities.
    - Keep TRAIN disjoint from VAL/TEST on the chosen entity axis.
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


def cold_start_split_by_entity(
    df: pd.DataFrame,
    entity_col: str,
    test_entity_frac: float,
    val_entity_frac: float,
    seed: int,

) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (train_df, val_df, test_df, val_entities, test_entities).

    val_entities and test_entities are sampled from unique entities.
    val_df and test_df contain *all* rows for the selected entities.
    train_df contains all remaining interactions.
    """

    if not (0.0 < test_entity_frac < 1.0):
        raise ValueError("test_entity_frac must be in (0, 1)")
    if not (0.0 < val_entity_frac < 1.0):
        raise ValueError("val_entity_frac must be in (0, 1)")
    if test_entity_frac + val_entity_frac >= 1.0:
        raise ValueError("test_entity_frac + val_entity_frac must be < 1.0")
    if entity_col not in df.columns:
        raise KeyError(f"entity_col '{entity_col}' not found in dataframe columns")

    # Ensure consistent indexing
    df = df.reset_index(drop=True)

    unique_entities = pd.Series(df[entity_col].dropna().unique())
    if len(unique_entities) == 0:
        raise ValueError("No entities found (unique_entities is empty). Check entity_col.")

    n_test = max(1, int(round(len(unique_entities) * test_entity_frac)))
    test_entities = unique_entities.sample(n=n_test, random_state=seed).reset_index(drop=True)

    remain_entities = unique_entities[~unique_entities.isin(set(test_entities.tolist()))].reset_index(drop=True)
    if len(remain_entities) == 0:
        raise ValueError("No entities left after selecting test entities.")

    n_val = max(1, int(round(len(unique_entities) * val_entity_frac)))
    n_val = min(n_val, len(remain_entities))
    val_entities = remain_entities.sample(n=n_val, random_state=seed + 1).reset_index(drop=True)

    is_test = df[entity_col].isin(set(test_entities.tolist()))
    is_val = df[entity_col].isin(set(val_entities.tolist()))

    test_df = df[is_test].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    remain_df = df[~(is_test | is_val)].reset_index(drop=True)

    if len(remain_df) == 0:
        raise ValueError(
            "After selecting val/test entities, no interactions remain for train. "
            "Reduce val/test fractions or check the dataset."
        )

    train_df = remain_df.reset_index(drop=True)

    return train_df, val_df, test_df, val_entities, test_entities


def main() -> None:
    p = argparse.ArgumentParser(description="Create cold-start splits from existing positive CSV splits")
    p.add_argument("--train", required=True, type=Path, help="Path to current train CSV")
    p.add_argument("--valid", required=True, type=Path, help="Path to current valid CSV")
    p.add_argument("--test", required=True, type=Path, help="Path to current test CSV")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory")
    p.add_argument("--prefix", default="db", help="Output file prefix")
    p.add_argument("--entity_type", choices=["drug", "protein"], default="drug",
                   help="Entity axis for cold-start splitting")

    p.add_argument(
        "--entity_col",
        default=None,
        help="Entity ID column name. If omitted, inferred from entity_type.",
    )
    p.add_argument(
        "--prot_col",
        default=None,
        help="Protein ID column name (for auto dedupe subset inference and logging).",
    )
    p.add_argument(
        "--drug_col",
        default=None,
        help="Drug ID column name (for auto dedupe subset inference and logging).",
    )

    p.add_argument("--test_entity_frac", type=float, default=0.10,
                   help="Fraction of unique entities in test")
    p.add_argument("--val_entity_frac", type=float, default=0.10,
                   help="Fraction of unique entities in validation")
    p.add_argument("--seed", type=int, default=2, help="Random seed")
    p.add_argument(
        "--dedupe_subset",
        default=None,
        help=(
            "Optional comma-separated columns for duplicate removal (e.g. 'drug_id,uniprot_id'). "
            "If omitted, attempts pair-wise dedupe with inferred drug/protein columns."
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
        ["drugbank_id", "drug_id", "Drug", "0"],
    )
    inferred_prot_col = args.prot_col or _pick_first_existing_col(
        df,
        ["uniprot_id", "target_uniprot", "Target", "1"],
    )

    if args.entity_col:
        inferred_entity_col = args.entity_col
    elif args.entity_type == "drug":
        inferred_entity_col = inferred_drug_col
    else:
        inferred_entity_col = inferred_prot_col

    if inferred_entity_col is None:
        raise SystemExit(f"Could not infer entity_col for entity_type='{args.entity_type}'. Available columns: {list(df.columns)}")

    # De-duplicate
    if args.dedupe_subset:
        subset_cols = [c.strip() for c in args.dedupe_subset.split(",") if c.strip()]
        missing = [c for c in subset_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"dedupe_subset columns not found: {missing}. Available: {list(df.columns)}")
        df = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
    else:
        if inferred_drug_col is not None and inferred_prot_col is not None:
            df = df.drop_duplicates(subset=[inferred_drug_col, inferred_prot_col], keep="first").reset_index(drop=True)
        else:
            df = df.drop_duplicates(keep="first").reset_index(drop=True)

    train_out, val_out, test_out, val_entities, test_entities = cold_start_split_by_entity(
        df=df,
        entity_col=inferred_entity_col,
        test_entity_frac=args.test_entity_frac,
        val_entity_frac=args.val_entity_frac,
        seed=args.seed,
    )

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / f"{args.prefix}_pos_train.csv"
    val_path = args.out_dir / f"{args.prefix}_pos_val.csv"
    test_path = args.out_dir / f"{args.prefix}_pos_test.csv"
    val_entities_path = args.out_dir / f"{args.prefix}_val_{args.entity_type}s.txt"
    test_entities_path = args.out_dir / f"{args.prefix}_test_{args.entity_type}s.txt"

    def count_pos_neg(df, label_col="interaction"):
        if label_col in df.columns:
            return df[label_col].value_counts().sort_index()
        return pd.Series(dtype="int64")

    print("Train:")
    print(count_pos_neg(train_out))

    print("\nVal:")
    print(count_pos_neg(val_out))

    print("\nTest:")
    print(count_pos_neg(test_out))

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    val_entities_path.write_text("\n".join(map(str, val_entities.tolist())) + "\n")
    test_entities_path.write_text("\n".join(map(str, test_entities.tolist())) + "\n")

    # Print a short summary
    print(f"=== Cold-start ({args.entity_type}) split created ===")
    print(f"Input rows (after dedupe): {len(df):,}")
    print(f"Unique {args.entity_type}s: {df[inferred_entity_col].nunique():,}")
    print(f"Val {args.entity_type}s: {len(val_entities):,} ({args.val_entity_frac:.2%} of unique {args.entity_type}s)")
    print(f"Test {args.entity_type}s: {len(test_entities):,} ({args.test_entity_frac:.2%} of unique {args.entity_type}s)")
    print(f"Rows -> train: {len(train_out):,} | val: {len(val_out):,} | test: {len(test_out):,}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {val_entities_path}")
    print(f"Saved: {test_entities_path}")


if __name__ == "__main__":
    main()
