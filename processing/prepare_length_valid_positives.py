#!/usr/bin/env python3
"""Prepare a positive-only DTI CSV with length and embedding-validity constraints.

Rules applied
-------------
1. Keep positive interactions only (label == 1 if a label column exists).
2. Keep rows with non-empty SMILES and Target_sequence.
3. Keep rows with SMILES length <= max_drug_len.
4. Keep rows with Target_sequence length <= max_protein_len.
5. Keep rows where both file-based embeddings exist on disk:
   - protein: {protein_dir}/{uniprot_id} contains at least one file
   - drug: {drug_dir}/{drug_id}_unimol.pt OR {drug_dir}/{drug_id}.pt exists
6. Deduplicate by (drug_id, uniprot_id).

Output schema
-------------
drug_id, uniprot_id, SMILES, Target_sequence, interaction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _detect_col(df: pd.DataFrame, candidates: list[str], friendly_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {friendly_name}. Tried: {candidates}")


def _is_nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _protein_embedding_exists(uniprot_id: str, protein_dir: Path) -> bool:
    folder = protein_dir / str(uniprot_id)
    if not folder.is_dir():
        return False
    try:
        return any(p.is_file() for p in folder.iterdir())
    except Exception:
        return False


def _drug_embedding_exists(drug_id: str, drug_dir: Path) -> bool:
    did = str(drug_id)
    return (drug_dir / f"{did}_unimol.pt").exists() or (drug_dir / f"{did}.pt").exists()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare positive-only, length-constrained, embedding-valid DTI CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--max_drug_len", type=int, default=400)
    p.add_argument("--max_protein_len", type=int, default=1000)
    p.add_argument("--protein_dir", type=Path, default=Path("embeddings"))
    p.add_argument("--drug_dir", type=Path, default=Path("drug/embeddings_atomic"))
    p.add_argument("--dropped_csv", type=Path, default=None,
                   help="Optional CSV for dropped rows with boolean rule columns")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)

    drug_col = _detect_col(df, ["drug_id", "0"], "drug id column")
    prot_col = _detect_col(df, ["uniprot_id", "1"], "protein id column")
    smiles_col = _detect_col(df, ["SMILES", "smiles"], "SMILES column")
    seq_col = _detect_col(df, ["Target_sequence", "Target sequence", "sequence"], "Target_sequence column")
    label_col = None
    for c in ["interaction", "Label", "label"]:
        if c in df.columns:
            label_col = c
            break

    n0 = len(df)

    # Positive-only (if label exists)
    if label_col is not None:
        label_vals = pd.to_numeric(df[label_col], errors="coerce")
        keep_positive = label_vals.eq(1)
    else:
        keep_positive = pd.Series(True, index=df.index)

    smiles_nonempty = _is_nonempty(df[smiles_col])
    seq_nonempty = _is_nonempty(df[seq_col])

    smiles_len = df[smiles_col].astype(str).str.len()
    seq_len = df[seq_col].astype(str).str.len()

    keep_drug_len = smiles_len.le(args.max_drug_len)
    keep_prot_len = seq_len.le(args.max_protein_len)

    # Embedding existence checks are mapped from unique IDs for efficiency.
    drug_ids = df[drug_col].astype(str)
    prot_ids = df[prot_col].astype(str)

    uniq_drugs = sorted(drug_ids.unique().tolist())
    uniq_prots = sorted(prot_ids.unique().tolist())

    drug_ok_map = {d: _drug_embedding_exists(d, args.drug_dir) for d in uniq_drugs}
    prot_ok_map = {u: _protein_embedding_exists(u, args.protein_dir) for u in uniq_prots}

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

    out_df = df.loc[keep_mask, [drug_col, prot_col, smiles_col, seq_col]].copy()
    out_df = out_df.rename(
        columns={
            drug_col: "drug_id",
            prot_col: "uniprot_id",
            smiles_col: "SMILES",
            seq_col: "Target_sequence",
        }
    )
    out_df["interaction"] = 1

    before_dedupe = len(out_df)
    out_df = out_df.drop_duplicates(subset=["drug_id", "uniprot_id"], keep="first").reset_index(drop=True)
    deduped = before_dedupe - len(out_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print("=== Prepared Positive Dataset ===")
    print(f"input_csv: {args.input_csv}")
    print(f"output_csv: {args.output_csv}")
    print(f"rows_in: {n0:,}")
    print(f"rows_kept_before_dedupe: {before_dedupe:,}")
    print(f"duplicates_removed_on_pair: {deduped:,}")
    print(f"rows_out: {len(out_df):,}")
    print(f"max_drug_len: {args.max_drug_len}")
    print(f"max_protein_len: {args.max_protein_len}")
    print("Rule failures (counts over original rows):")
    print(f"  non_positive: {(~keep_positive).sum():,}")
    print(f"  empty_smiles: {(~smiles_nonempty).sum():,}")
    print(f"  empty_sequence: {(~seq_nonempty).sum():,}")
    print(f"  drug_len_exceeded: {(~keep_drug_len).sum():,}")
    print(f"  protein_len_exceeded: {(~keep_prot_len).sum():,}")
    print(f"  missing_drug_embedding: {(~keep_drug_emb).sum():,}")
    print(f"  missing_protein_embedding: {(~keep_prot_emb).sum():,}")

    if args.dropped_csv is not None:
        dropped = df.loc[~keep_mask].copy()
        dropped["_keep_positive"] = keep_positive[~keep_mask]
        dropped["_smiles_nonempty"] = smiles_nonempty[~keep_mask]
        dropped["_sequence_nonempty"] = seq_nonempty[~keep_mask]
        dropped["_keep_drug_len"] = keep_drug_len[~keep_mask]
        dropped["_keep_prot_len"] = keep_prot_len[~keep_mask]
        dropped["_keep_drug_emb"] = keep_drug_emb[~keep_mask]
        dropped["_keep_prot_emb"] = keep_prot_emb[~keep_mask]
        args.dropped_csv.parent.mkdir(parents=True, exist_ok=True)
        dropped.to_csv(args.dropped_csv, index=False)
        print(f"dropped_csv: {args.dropped_csv} ({len(dropped):,} rows)")


if __name__ == "__main__":
    main()
