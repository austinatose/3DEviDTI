#!/usr/bin/env python3
"""Cold-start splitting utility.

Goal:
  - Read existing train/val/test CSVs (or a single input CSV), collate into one
    dataset (drop duplicates).
  - Create a cold-start split by drug or protein, where TEST and VAL each contain
    all interactions for disjoint sets of entities.
  - Keep TRAIN disjoint from VAL/TEST on the chosen entity axis.

Split methods:
  - random:         randomly sample entities for val/test (original behavior)
  - median_cluster: cluster entities by chemical/sequence similarity, then select
                    median-sized clusters for val/test (avoids extreme entities)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
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


def cold_start_split_by_median_clusters(
    df: pd.DataFrame,
    entity_col: str,
    entity_type: str,
    smiles_col: str = "SMILES",
    sequence_col: str = "Target_sequence",
    test_entity_frac: float = 0.10,
    val_entity_frac: float = 0.10,
    drug_threshold: float = 0.4,
    protein_threshold: float = 0.5,
    protein_kmer_k: int = 3,
    seed: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Cold-start split selecting median-ranked clusters for val/test.

    Instead of randomly sampling entities:
    1. Clusters entities by chemical/sequence similarity
    2. Ranks clusters by size (ascending)
    3. Assigns clusters nearest the median rank to val/test
    4. Extreme clusters (tiny singletons and huge families) stay in train
    """
    try:
        from processing.similarity_split import cluster_drugs, cluster_proteins
    except ImportError:
        from similarity_split import cluster_drugs, cluster_proteins

    if not (0.0 < test_entity_frac < 1.0):
        raise ValueError("test_entity_frac must be in (0, 1)")
    if not (0.0 < val_entity_frac < 1.0):
        raise ValueError("val_entity_frac must be in (0, 1)")
    if test_entity_frac + val_entity_frac >= 1.0:
        raise ValueError("test_entity_frac + val_entity_frac must be < 1.0")
    if entity_col not in df.columns:
        raise KeyError(f"entity_col '{entity_col}' not found in dataframe columns")

    df = df.reset_index(drop=True)

    # --- Cluster entities ---
    if entity_type == "drug":
        if smiles_col not in df.columns:
            raise KeyError(f"smiles_col '{smiles_col}' not in dataframe columns: {list(df.columns)}")
        unique_ents = df[[entity_col, smiles_col]].drop_duplicates(subset=[entity_col])
        print(f"Clustering {len(unique_ents)} unique drugs (Tanimoto distance threshold={drug_threshold}) ...")
        labels = cluster_drugs(unique_ents[smiles_col].tolist(), threshold=drug_threshold)
    else:
        if sequence_col not in df.columns:
            raise KeyError(f"sequence_col '{sequence_col}' not in dataframe columns: {list(df.columns)}")
        unique_ents = df[[entity_col, sequence_col]].drop_duplicates(subset=[entity_col])
        print(f"Clustering {len(unique_ents)} unique proteins (Jaccard threshold={protein_threshold}, k={protein_kmer_k}) ...")
        labels = cluster_proteins(unique_ents[sequence_col].tolist(), threshold=protein_threshold, k=protein_kmer_k)

    entity_ids = unique_ents[entity_col].values
    n_entities = len(entity_ids)

    # --- Build cluster -> entities mapping ---
    cluster_map: dict[int, list] = {}
    for eid, lab in zip(entity_ids, labels):
        cluster_map.setdefault(int(lab), []).append(eid)

    n_clusters = len(cluster_map)
    print(f"  -> {n_clusters} clusters from {n_entities} unique {entity_type}s")

    # --- Sort clusters by size (ascending) ---
    clusters_by_size = sorted(cluster_map.items(), key=lambda x: len(x[1]))
    cluster_sizes = [len(c[1]) for c in clusters_by_size]

    median_size = int(np.median(cluster_sizes))
    print(
        f"  Cluster sizes: min={min(cluster_sizes)}, "
        f"median={median_size}, max={max(cluster_sizes)}, "
        f"mean={np.mean(cluster_sizes):.1f}"
    )

    # --- Select median-ranked clusters for val/test ---
    # Order cluster indices by proximity to the median rank
    median_rank = n_clusters // 2
    indices_by_median = sorted(
        range(n_clusters),
        key=lambda i: (abs(i - median_rank), i),
    )

    n_val_target = max(1, int(round(n_entities * val_entity_frac)))
    n_test_target = max(1, int(round(n_entities * test_entity_frac)))

    val_entities: list = []
    test_entities: list = []
    selected_indices: set[int] = set()

    for idx in indices_by_median:
        if len(val_entities) >= n_val_target and len(test_entities) >= n_test_target:
            break

        cluster_ents = clusters_by_size[idx][1]

        val_need = n_val_target - len(val_entities)
        test_need = n_test_target - len(test_entities)

        if val_need > 0 and (test_need <= 0 or val_need >= test_need):
            val_entities.extend(cluster_ents)
        elif test_need > 0:
            test_entities.extend(cluster_ents)
        else:
            break

        selected_indices.add(idx)

    # --- Partition rows ---
    val_set = set(val_entities)
    test_set = set(test_entities)

    is_val = df[entity_col].isin(val_set)
    is_test = df[entity_col].isin(test_set)

    val_df = df[is_val].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)
    train_df = df[~(is_val | is_test)].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(
            "After selecting val/test clusters, no interactions remain for train. "
            "Reduce val/test fractions or check the dataset."
        )

    # --- Summary ---
    sel_sizes = sorted(cluster_sizes[i] for i in selected_indices)
    train_cluster_sizes = sorted(cluster_sizes[i] for i in range(n_clusters) if i not in selected_indices)
    print(
        f"  Val/test: {len(selected_indices)} clusters selected "
        f"(sizes {sel_sizes[0]}..{sel_sizes[-1]}, "
        f"val_entities={len(val_entities)}, test_entities={len(test_entities)})"
    )
    print(
        f"  Train:    {n_clusters - len(selected_indices)} clusters "
        f"(sizes {train_cluster_sizes[0]}..{train_cluster_sizes[-1]})"
    )

    val_entities_series = pd.Series(val_entities).reset_index(drop=True)
    test_entities_series = pd.Series(test_entities).reset_index(drop=True)

    return train_df, val_df, test_df, val_entities_series, test_entities_series


def main() -> None:
    p = argparse.ArgumentParser(description="Create cold-start splits from existing positive CSV splits")

    # --- Input (either --input OR --train/--valid/--test) ---
    p.add_argument("--input", type=Path, default=None,
                   help="Single input CSV (alternative to --train/--valid/--test)")
    p.add_argument("--train", type=Path, default=None, help="Path to current train CSV")
    p.add_argument("--valid", type=Path, default=None, help="Path to current valid CSV")
    p.add_argument("--test", type=Path, default=None, help="Path to current test CSV")

    p.add_argument("--out_dir", required=True, type=Path, help="Output directory")
    p.add_argument("--prefix", default="db", help="Output file prefix")
    p.add_argument("--entity_type", choices=["drug", "protein"], default="drug",
                   help="Entity axis for cold-start splitting")

    p.add_argument(
        "--entity_col", default=None,
        help="Entity ID column name. If omitted, inferred from entity_type.",
    )
    p.add_argument(
        "--prot_col", default=None,
        help="Protein ID column name (for auto dedupe subset inference and logging).",
    )
    p.add_argument(
        "--drug_col", default=None,
        help="Drug ID column name (for auto dedupe subset inference and logging).",
    )

    p.add_argument("--test_entity_frac", type=float, default=0.10,
                   help="Fraction of unique entities in test")
    p.add_argument("--val_entity_frac", type=float, default=0.10,
                   help="Fraction of unique entities in validation")
    p.add_argument("--seed", type=int, default=2, help="Random seed")
    p.add_argument(
        "--dedupe_subset", default=None,
        help=(
            "Optional comma-separated columns for duplicate removal (e.g. 'drug_id,uniprot_id'). "
            "If omitted, attempts pair-wise dedupe with inferred drug/protein columns."
        ),
    )

    # --- Split method ---
    p.add_argument(
        "--split_method", choices=["random", "median_cluster"], default="random",
        help="'random' samples entities randomly; 'median_cluster' uses similarity "
             "clustering and selects median-ranked clusters for val/test",
    )
    p.add_argument("--smiles_col", default="SMILES",
                   help="SMILES column name (for drug clustering)")
    p.add_argument("--sequence_col", default="Target_sequence",
                   help="Sequence column name (for protein clustering)")
    p.add_argument("--drug_threshold", type=float, default=0.4,
                   help="Tanimoto distance cutoff for drug Butina clustering")
    p.add_argument("--protein_threshold", type=float, default=0.5,
                   help="Jaccard similarity cutoff for protein clustering")

    args = p.parse_args()

    # --- Load input ---
    if args.input is not None:
        df = pd.read_csv(args.input)
    elif args.train and args.valid and args.test:
        train_df = pd.read_csv(args.train)
        valid_df = pd.read_csv(args.valid)
        test_df = pd.read_csv(args.test)
        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    else:
        raise SystemExit("Provide either --input or all of --train/--valid/--test")

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

    # --- Split ---
    if args.split_method == "median_cluster":
        train_out, val_out, test_out, val_entities, test_entities = cold_start_split_by_median_clusters(
            df=df,
            entity_col=inferred_entity_col,
            entity_type=args.entity_type,
            smiles_col=args.smiles_col,
            sequence_col=args.sequence_col,
            test_entity_frac=args.test_entity_frac,
            val_entity_frac=args.val_entity_frac,
            drug_threshold=args.drug_threshold,
            protein_threshold=args.protein_threshold,
            seed=args.seed,
        )
    else:
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
    print(f"\n=== Cold-start ({args.entity_type}, method={args.split_method}) split created ===")
    print(f"Input rows (after dedupe): {len(df):,}")
    print(f"Unique {args.entity_type}s: {df[inferred_entity_col].nunique():,}")
    print(f"Val {args.entity_type}s: {len(val_entities):,} ({args.val_entity_frac:.2%} target)")
    print(f"Test {args.entity_type}s: {len(test_entities):,} ({args.test_entity_frac:.2%} target)")
    print(f"Rows -> train: {len(train_out):,} | val: {len(val_out):,} | test: {len(test_out):,}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {val_entities_path}")
    print(f"Saved: {test_entities_path}")


if __name__ == "__main__":
    main()
