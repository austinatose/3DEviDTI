#!/usr/bin/env python3
"""Build constrained DTI train/val/test splits with negatives in one command.

Pipeline
--------
1. Load an input CSV and keep positive interactions only.
2. Apply constraints:
   - non-empty SMILES and Target_sequence
   - max SMILES length <= max_drug_len
   - max Target_sequence length <= max_protein_len
   - file-based protein and drug embeddings must exist
3. Deduplicate by (drug_id, uniprot_id).
4. Similarity-aware split (drug/protein/both).
5. Generate synthetic negatives inside each split's entity universe.
6. Save outputs and validate schema + length bounds + embedding existence.

Example
-------
python processing/build_constrained_splits.py \
  --input_csv ../MocFormer/DrugBank_uni_esm2_3B_pos_with_file_emb.csv \
  --out_dir ../MocFormer \
  --base_name DrugBank_uni_esm2_3B_pos_len_d400_p1000 \
  --max_drug_len 400 \
  --max_protein_len 1000 \
  --protein_dir embeddings \
  --drug_dir drug/embeddings_atomic \
  --mode protein \
  --neg_ratio 1.0 \
  --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

try:
    from processing.similarity_split import similarity_split
except Exception:
    from similarity_split import similarity_split


def detect_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing {label}. Tried columns: {candidates}")


def is_nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def protein_embedding_exists(uniprot_id: str, protein_dir: Path) -> bool:
    folder = protein_dir / str(uniprot_id)
    if not folder.is_dir():
        return False
    try:
        return any(p.is_file() for p in folder.iterdir())
    except Exception:
        return False


def drug_embedding_exists(drug_id: str, drug_dir: Path) -> bool:
    did = str(drug_id)
    return (drug_dir / f"{did}_unimol.pt").exists() or (drug_dir / f"{did}.pt").exists()


def prepare_positives(
    input_csv: Path,
    max_drug_len: int,
    max_protein_len: int,
    protein_dir: Path,
    drug_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(input_csv)

    drug_col = detect_col(df, ["drug_id", "0"], "drug id column")
    prot_col = detect_col(df, ["uniprot_id", "1"], "protein id column")
    smiles_col = detect_col(df, ["SMILES", "smiles"], "SMILES column")
    seq_col = detect_col(df, ["Target_sequence", "Target sequence", "sequence"], "Target sequence column")

    label_col = None
    for c in ["interaction", "Label", "label"]:
        if c in df.columns:
            label_col = c
            break

    n0 = len(df)

    if label_col is not None:
        label_numeric = pd.to_numeric(df[label_col], errors="coerce")
        keep_positive = label_numeric.eq(1)
    else:
        keep_positive = pd.Series(True, index=df.index)

    smiles_nonempty = is_nonempty(df[smiles_col])
    seq_nonempty = is_nonempty(df[seq_col])

    keep_drug_len = df[smiles_col].astype(str).str.len().le(max_drug_len)
    keep_prot_len = df[seq_col].astype(str).str.len().le(max_protein_len)

    drug_ids = df[drug_col].astype(str)
    prot_ids = df[prot_col].astype(str)

    uniq_drugs = sorted(drug_ids.unique().tolist())
    uniq_prots = sorted(prot_ids.unique().tolist())

    drug_ok_map = {d: drug_embedding_exists(d, drug_dir) for d in uniq_drugs}
    prot_ok_map = {u: protein_embedding_exists(u, protein_dir) for u in uniq_prots}

    keep_drug_emb = drug_ids.map(drug_ok_map)
    keep_prot_emb = prot_ids.map(prot_ok_map)

    keep_mask = (
        keep_positive
        & smiles_nonempty
        & seq_nonempty
        & keep_drug_len
        & keep_prot_len
        & keep_drug_emb
        & keep_prot_emb
    )

    kept = df.loc[keep_mask, [drug_col, prot_col, smiles_col, seq_col]].copy()
    kept = kept.rename(
        columns={
            drug_col: "drug_id",
            prot_col: "uniprot_id",
            smiles_col: "SMILES",
            seq_col: "Target_sequence",
        }
    )
    kept["interaction"] = 1

    before_dedupe = len(kept)
    kept = kept.drop_duplicates(subset=["drug_id", "uniprot_id"], keep="first").reset_index(drop=True)

    dropped = df.loc[~keep_mask].copy()
    dropped["_keep_positive"] = keep_positive[~keep_mask]
    dropped["_smiles_nonempty"] = smiles_nonempty[~keep_mask]
    dropped["_sequence_nonempty"] = seq_nonempty[~keep_mask]
    dropped["_keep_drug_len"] = keep_drug_len[~keep_mask]
    dropped["_keep_prot_len"] = keep_prot_len[~keep_mask]
    dropped["_keep_drug_emb"] = keep_drug_emb[~keep_mask]
    dropped["_keep_prot_emb"] = keep_prot_emb[~keep_mask]

    stats = {
        "rows_in": int(n0),
        "rows_kept_before_dedupe": int(before_dedupe),
        "duplicates_removed_on_pair": int(before_dedupe - len(kept)),
        "rows_out": int(len(kept)),
        "non_positive": int((~keep_positive).sum()),
        "empty_smiles": int((~smiles_nonempty).sum()),
        "empty_sequence": int((~seq_nonempty).sum()),
        "drug_len_exceeded": int((~keep_drug_len).sum()),
        "protein_len_exceeded": int((~keep_prot_len).sum()),
        "missing_drug_embedding": int((~keep_drug_emb).sum()),
        "missing_protein_embedding": int((~keep_prot_emb).sum()),
    }

    return kept, dropped, stats


def sample_negatives_for_split(
    pos_df: pd.DataFrame,
    all_positive_pairs: set[tuple[str, str]],
    ratio: float,
    seed: int,
    max_attempts_multiplier: int = 20,
) -> pd.DataFrame:
    rng = random.Random(seed)

    drugs = pos_df["drug_id"].astype(str).unique().tolist()
    prots = pos_df["uniprot_id"].astype(str).unique().tolist()

    n_target = int(round(len(pos_df) * ratio))
    negatives: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    attempts = 0
    max_attempts = max(n_target * max_attempts_multiplier, 1)
    while len(negatives) < n_target and attempts < max_attempts:
        pair = (rng.choice(drugs), rng.choice(prots))
        if pair not in all_positive_pairs and pair not in seen:
            seen.add(pair)
            negatives.append(pair)
        attempts += 1

    if len(negatives) < n_target:
        print(
            f"  Warning: requested {n_target} negatives, produced {len(negatives)} "
            f"after {attempts} attempts"
        )

    lookup_drug_smiles = dict(zip(pos_df["drug_id"], pos_df["SMILES"]))
    lookup_prot_seq = dict(zip(pos_df["uniprot_id"], pos_df["Target_sequence"]))

    rows = []
    for d, p in negatives:
        rows.append(
            {
                "drug_id": d,
                "uniprot_id": p,
                "SMILES": lookup_drug_smiles.get(d, ""),
                "Target_sequence": lookup_prot_seq.get(p, ""),
                "interaction": 0,
            }
        )

    return pd.DataFrame(rows)


def validate_output_csv(
    csv_path: Path,
    max_drug_len: int,
    max_protein_len: int,
    protein_dir: Path,
    drug_dir: Path,
) -> None:
    df = pd.read_csv(csv_path)

    required = ["drug_id", "uniprot_id", "SMILES", "Target_sequence", "interaction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")

    max_smi = int(df["SMILES"].astype(str).str.len().max()) if len(df) else 0
    max_seq = int(df["Target_sequence"].astype(str).str.len().max()) if len(df) else 0

    if max_smi > max_drug_len:
        raise ValueError(f"{csv_path}: max SMILES length {max_smi} > {max_drug_len}")
    if max_seq > max_protein_len:
        raise ValueError(f"{csv_path}: max Target_sequence length {max_seq} > {max_protein_len}")

    uniq_drugs = sorted(df["drug_id"].astype(str).unique().tolist())
    uniq_prots = sorted(df["uniprot_id"].astype(str).unique().tolist())

    missing_drugs = [d for d in uniq_drugs if not drug_embedding_exists(d, drug_dir)]
    missing_prots = [p for p in uniq_prots if not protein_embedding_exists(p, protein_dir)]

    if missing_drugs:
        raise ValueError(f"{csv_path}: missing drug embeddings for {len(missing_drugs)} ids")
    if missing_prots:
        raise ValueError(f"{csv_path}: missing protein embeddings for {len(missing_prots)} ids")

    print(
        f"Validated {csv_path.name}: rows={len(df):,}, "
        f"labels={df['interaction'].value_counts(dropna=False).to_dict()}, "
        f"max_smiles_len={max_smi}, max_protein_len={max_seq}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build constrained similarity splits with negatives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--base_name", type=str, required=True)

    p.add_argument("--max_drug_len", type=int, default=400)
    p.add_argument("--max_protein_len", type=int, default=1000)
    p.add_argument("--protein_dir", type=Path, default=Path("embeddings"))
    p.add_argument("--drug_dir", type=Path, default=Path("drug/embeddings_atomic"))

    p.add_argument("--mode", choices=["drug", "protein", "both"], default="protein")
    p.add_argument("--drug_threshold", type=float, default=0.4)
    p.add_argument("--protein_threshold", type=float, default=0.5)
    p.add_argument("--protein_kmer_k", type=int, default=3)
    p.add_argument("--protein_cluster_method", choices=["jaccard", "mmseqs2"], default="jaccard")
    p.add_argument("--split_assignment", choices=["greedy", "distance_max"], default="greedy")
    p.add_argument("--cluster_distance_agg", choices=["max", "mean"], default="max")
    p.add_argument("--distance_refine_iters", type=int, default=0)
    p.add_argument("--distance_size_tolerance", type=float, default=0.02)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    p.add_argument("--neg_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dropped", action="store_true")

    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_valid_csv = out_dir / f"{args.base_name}_valid.csv"
    dropped_csv = out_dir / f"{args.base_name}_dropped.csv"

    print("Step 1/4: Preparing constrained valid positives...")
    pos_df, dropped_df, prep_stats = prepare_positives(
        input_csv=args.input_csv,
        max_drug_len=args.max_drug_len,
        max_protein_len=args.max_protein_len,
        protein_dir=args.protein_dir,
        drug_dir=args.drug_dir,
    )
    pos_df.to_csv(pos_valid_csv, index=False)
    if args.save_dropped:
        dropped_df.to_csv(dropped_csv, index=False)

    print("Prepared positives summary:")
    for k, v in prep_stats.items():
        print(f"  {k}: {v:,}")
    print(f"  saved: {pos_valid_csv}")
    if args.save_dropped:
        print(f"  saved dropped rows: {dropped_csv}")

    print("Step 2/4: Similarity split...")
    split_prefix = f"{args.base_name}_sim_{args.mode}"
    train_pos, val_pos, test_pos, split_stats = similarity_split(
        pos_df,
        mode=args.mode,
        drug_col="drug_id",
        smiles_col="SMILES",
        protein_col="uniprot_id",
        sequence_col="Target_sequence",
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

    train_pos_csv = out_dir / f"{split_prefix}_train.csv"
    val_pos_csv = out_dir / f"{split_prefix}_val.csv"
    test_pos_csv = out_dir / f"{split_prefix}_test.csv"

    train_pos.to_csv(train_pos_csv, index=False)
    val_pos.to_csv(val_pos_csv, index=False)
    test_pos.to_csv(test_pos_csv, index=False)

    print(f"  saved: {train_pos_csv}")
    print(f"  saved: {val_pos_csv}")
    print(f"  saved: {test_pos_csv}")

    print("Step 3/4: Negative generation...")
    all_pos = pd.concat([train_pos, val_pos, test_pos], ignore_index=True)
    all_positive_pairs = set(zip(all_pos["drug_id"].astype(str), all_pos["uniprot_id"].astype(str)))

    split_tuples = [
        ("train", train_pos, args.seed),
        ("val", val_pos, args.seed + 1),
        ("test", test_pos, args.seed + 2),
    ]

    neg_prefix = f"{split_prefix}_neg"
    final_paths = {}
    for split_name, pos_split, split_seed in split_tuples:
        neg_df = sample_negatives_for_split(
            pos_df=pos_split,
            all_positive_pairs=all_positive_pairs,
            ratio=args.neg_ratio,
            seed=split_seed,
        )

        keep_cols = ["drug_id", "uniprot_id", "SMILES", "Target_sequence", "interaction"]
        combined = pd.concat([pos_split[keep_cols], neg_df[keep_cols]], ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)

        out_path = out_dir / f"{neg_prefix}_{split_name}.csv"
        combined.to_csv(out_path, index=False)
        final_paths[split_name] = out_path

        n_pos = int((combined["interaction"] == 1).sum())
        n_neg = int((combined["interaction"] == 0).sum())
        print(f"  {split_name}: saved {out_path.name} rows={len(combined):,} pos={n_pos:,} neg={n_neg:,}")

    print("Step 4/4: Final validation...")
    for split_name in ["train", "val", "test"]:
        validate_output_csv(
            final_paths[split_name],
            max_drug_len=args.max_drug_len,
            max_protein_len=args.max_protein_len,
            protein_dir=args.protein_dir,
            drug_dir=args.drug_dir,
        )

    print("Done.")
    print("Final files:")
    for split_name in ["train", "val", "test"]:
        print(f"  {final_paths[split_name]}")


if __name__ == "__main__":
    main()
