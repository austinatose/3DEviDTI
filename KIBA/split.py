from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Ensure project root is importable when running from this subfolder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.similarity_split import similarity_split


def print_label_balance(df: pd.DataFrame, label_col: str, name: str) -> None:
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    print(f"\n=== Label balance for {name} ===")
    print(f"Total samples: {total}")
    print(f"Negatives (0): {neg} ({neg / total:.3f})")
    print(f"Positives (1): {pos} ({pos / total:.3f})")


def build_output_path(out_dir: Path, split_name: str, suffix: str) -> Path:
    return out_dir / f"KIBA_pairs_{split_name}{suffix}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create KIBA train/val/test files with similarity-aware splitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs_csv", default="lists/KIBA/KIBA_pairs.csv")
    parser.add_argument("--out_dir", default="lists/KIBA")
    parser.add_argument("--output_suffix", default="_sim_protein")

    parser.add_argument("--mode", choices=["drug", "protein", "both"], default="protein")
    parser.add_argument("--drug_threshold", type=float, default=0.4)
    parser.add_argument("--protein_threshold", type=float, default=0.5)
    parser.add_argument("--protein_kmer_k", type=int, default=3)
    parser.add_argument("--protein_cluster_method", choices=["jaccard", "mmseqs2"], default="jaccard")
    parser.add_argument("--split_assignment", choices=["greedy", "distance_max"], default="greedy")
    parser.add_argument("--cluster_distance_agg", choices=["max", "mean"], default="max")
    parser.add_argument("--distance_refine_iters", type=int, default=0)
    parser.add_argument("--distance_size_tolerance", type=float, default=0.02)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = pd.read_csv(args.pairs_csv)

    train_df, val_df, test_df, stats = similarity_split(
        df,
        mode=args.mode,
        drug_col="drug_id",
        smiles_col="smiles",
        protein_col="uniprot_id",
        sequence_col="Target sequence",
        label_col="interaction",
        drug_threshold=args.drug_threshold,
        protein_threshold=args.protein_threshold,
        protein_kmer_k=args.protein_kmer_k,
        protein_cluster_method=args.protein_cluster_method,
        split_assignment=args.split_assignment,
        cluster_distance_agg=args.cluster_distance_agg,
        distance_refine_iters=args.distance_refine_iters,
        distance_size_tolerance=args.distance_size_tolerance,
        target_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = build_output_path(out_dir, "train", args.output_suffix)
    val_path = build_output_path(out_dir, "val", args.output_suffix)
    test_path = build_output_path(out_dir, "test", args.output_suffix)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")

    print(f"\nTotal rows: {len(df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print_label_balance(df, "interaction", "full")
    print_label_balance(train_df, "interaction", "train")
    print_label_balance(val_df, "interaction", "val")
    print_label_balance(test_df, "interaction", "test")

    print("\nSplit stats:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()