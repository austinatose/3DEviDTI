#!/usr/bin/env python3
"""Generate synthetic negative drug-target pairs for each split.

Strategy
--------
For each split (train / val / test):
  1. Take the drugs and proteins that appear in that split's positives.
  2. Sample random (drug, protein) pairs from their Cartesian product,
     excluding ALL known positive pairs (across the entire dataset).
  3. Look up the SMILES / sequence for each sampled pair from the positive
     files (so the output has the same columns).
  4. Write a combined file containing positives + negatives (interaction=0).

The negatives are drawn entirely within each split's drug/protein universe,
so the train/val/test entity boundaries from similarity splitting are preserved.

Usage
-----
    python processing/generate_negatives.py \
        --train  lists/db_new/db_pos_train.csv \
        --val    lists/db_new/db_pos_val.csv \
        --test   lists/db_new/db_pos_test.csv \
        --out_dir lists/db_new \
        --ratio 1.0 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


def load_splits(
    train_path: Path, val_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    return train, val, test


def build_lookup_tables(
    dfs: list[pd.DataFrame],
    drug_col: str,
    smiles_col: str,
    protein_col: str,
    sequence_col: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build drug→SMILES and protein→sequence dicts from all splits combined."""
    all_df = pd.concat(dfs, ignore_index=True)
    drug_to_smiles = dict(zip(all_df[drug_col], all_df[smiles_col]))
    prot_to_seq    = dict(zip(all_df[protein_col], all_df[sequence_col]))
    return drug_to_smiles, prot_to_seq


def sample_negatives(
    positive_df: pd.DataFrame,
    all_positives_set: set[tuple[str, str]],
    drug_to_smiles: dict[str, str],
    prot_to_seq: dict[str, str],
    drug_col: str,
    protein_col: str,
    smiles_col: str,
    sequence_col: str,
    label_col: str,
    ratio: float,
    seed: int,
    max_attempts_multiplier: int = 20,
) -> pd.DataFrame:
    """Sample negative pairs for one split.

    Parameters
    ----------
    positive_df : DataFrame
        The positive interactions for this split.
    all_positives_set : set of (drug_id, protein_id)
        ALL known positives across every split — used to filter candidates.
    ratio : float
        Number of negatives per positive (1.0 = balanced).
    max_attempts_multiplier : int
        Cap on sampling attempts = n_neg_wanted * this value.
    """
    rng = random.Random(seed)

    drugs    = list(positive_df[drug_col].unique())
    proteins = list(positive_df[protein_col].unique())
    n_neg_wanted = int(round(len(positive_df) * ratio))

    negatives: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    max_attempts = n_neg_wanted * max_attempts_multiplier
    attempts = 0

    while len(negatives) < n_neg_wanted and attempts < max_attempts:
        d = rng.choice(drugs)
        p = rng.choice(proteins)
        pair = (d, p)
        if pair not in all_positives_set and pair not in seen:
            negatives.append(pair)
            seen.add(pair)
        attempts += 1

    if len(negatives) < n_neg_wanted:
        print(
            f"  ⚠  Wanted {n_neg_wanted} negatives but only found {len(negatives)} "
            f"after {attempts} attempts. Consider increasing --ratio or reducing "
            "max_attempts_multiplier."
        )

    rows = []
    for d, p in negatives:
        rows.append(
            {
                drug_col:     d,
                protein_col:  p,
                smiles_col:   drug_to_smiles.get(d, ""),
                sequence_col: prot_to_seq.get(p, ""),
                label_col:    0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic negative pairs for DTI splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",   type=Path, required=True)
    parser.add_argument("--val",     type=Path, required=True)
    parser.add_argument("--test",    type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("lists/db_new"))
    parser.add_argument("--prefix",  type=str,  default="db")
    parser.add_argument("--ratio",   type=float, default=1.0,
                        help="Negatives per positive (1.0 = balanced)")
    parser.add_argument("--seed",    type=int,  default=42)

    # Column names
    parser.add_argument("--drug_col",     default="drug_id")
    parser.add_argument("--smiles_col",   default="SMILES")
    parser.add_argument("--protein_col",  default="uniprot_id")
    parser.add_argument("--sequence_col", default="Target_sequence")
    parser.add_argument("--label_col",    default="interaction")

    args = parser.parse_args()

    train_pos, val_pos, test_pos = load_splits(args.train, args.val, args.test)
    all_splits = [train_pos, val_pos, test_pos]

    drug_to_smiles, prot_to_seq = build_lookup_tables(
        all_splits, args.drug_col, args.smiles_col, args.protein_col, args.sequence_col
    )

    # Full set of known positives (used to avoid false negatives)
    all_positives_set: set[tuple[str, str]] = set()
    for df in all_splits:
        for d, p in zip(df[args.drug_col], df[args.protein_col]):
            all_positives_set.add((d, p))
    print(f"Total known positive pairs: {len(all_positives_set):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, pos_df, split_seed in [
        ("train", train_pos, args.seed),
        ("val",   val_pos,   args.seed + 1),
        ("test",  test_pos,  args.seed + 2),
    ]:
        print(f"\n[{split_name}]  {len(pos_df):,} positives → targeting {int(round(len(pos_df) * args.ratio)):,} negatives")

        neg_df = sample_negatives(
            positive_df=pos_df,
            all_positives_set=all_positives_set,
            drug_to_smiles=drug_to_smiles,
            prot_to_seq=prot_to_seq,
            drug_col=args.drug_col,
            protein_col=args.protein_col,
            smiles_col=args.smiles_col,
            sequence_col=args.sequence_col,
            label_col=args.label_col,
            ratio=args.ratio,
            seed=split_seed,
        )

        # Keep only the relevant columns from positives, then combine
        keep_cols = [args.drug_col, args.protein_col, args.smiles_col,
                     args.sequence_col, args.label_col]
        pos_out = pos_df[keep_cols].copy()
        combined = pd.concat([pos_out, neg_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=split_seed).reset_index(drop=True)

        out_path = args.out_dir / f"{args.prefix}_{split_name}.csv"
        combined.to_csv(out_path, index=False)

        n_pos = (combined[args.label_col] == 1).sum()
        n_neg = (combined[args.label_col] == 0).sum()
        print(f"  Saved {out_path}  ({len(combined):,} rows  pos={n_pos:,}  neg={n_neg:,})")


if __name__ == "__main__":
    main()
