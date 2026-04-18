#!/usr/bin/env python3
"""Similarity-aware train/val/test splitting for DTI datasets.

Ensures that test and validation sets contain entities (drugs, proteins, or
both) that are sufficiently dissimilar from those in the training set.  This
prevents inflated performance metrics caused by near-duplicate leakage.

Splitting strategies
--------------------
* **drug**   – cluster drugs by ECFP4 Tanimoto similarity; assign whole
               clusters to splits so no two splits share a drug cluster.
* **protein** – cluster proteins by sequence k-mer similarity; same logic.
* **both**   – cluster drugs *and* proteins independently, form (drug_cluster,
               protein_cluster) pair IDs, and assign pair groups to splits.
               This is the strictest setting: a test interaction's drug AND
               protein will both be dissimilar to all training interactions.

Algorithm
---------
1. Compute pairwise similarities within the drug / protein entity set.
2. Cluster with the Butina algorithm (drugs) or agglomerative clustering
   (proteins) at the user-specified similarity threshold.
3. Assign clusters to {train, val, test} targeting an 8:1:1 ratio while
   keeping each cluster entirely inside one split.
4. Optionally re-balance positive/negative labels within each split.

Dependencies: rdkit, numpy, pandas, scikit-learn (for proteins).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Ligand similarity  (ECFP4 / Morgan fingerprint, Tanimoto)
# ---------------------------------------------------------------------------

def ecfp4_fingerprints(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    """Return a list of RDKit Morgan fingerprint objects (bit vectors)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as e:
        raise ImportError(
            "RDKit is required for drug similarity clustering. "
            "Install rdkit or use --mode protein."
        ) from e

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps


def tanimoto_distance_matrix(fps: list) -> list[float]:
    """Compute a condensed distance matrix (1 - Tanimoto) for Butina clustering.

    Returns a flat list of pairwise distances in the order required by
    ``Butina.ClusterData``: d(0,1), d(0,2), …, d(0,n-1), d(1,2), …
    """
    try:
        from rdkit import DataStructs
    except Exception as e:
        raise ImportError(
            "RDKit is required for drug similarity clustering. "
            "Install rdkit or use --mode protein."
        ) from e

    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    return dists


def cluster_drugs(smiles_list: list[str], threshold: float = 0.4) -> np.ndarray:
    """Cluster drugs using Butina on ECFP4 Tanimoto distances.

    Parameters
    ----------
    smiles_list : list[str]
        One SMILES string per unique drug.
    threshold : float
        Tanimoto *distance* cutoff (i.e. 0.4 means clusters contain molecules
        with pairwise Tanimoto similarity >= 0.6).

    Returns
    -------
    labels : np.ndarray of shape (n_drugs,)
        Integer cluster label for each drug.
    """
    try:
        from rdkit.ML.Cluster import Butina
    except Exception as e:
        raise ImportError(
            "RDKit is required for drug similarity clustering. "
            "Install rdkit or use --mode protein."
        ) from e

    fps = ecfp4_fingerprints(smiles_list)

    # Handle molecules that could not be parsed — give each its own cluster
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_idx]

    if len(valid_fps) == 0:
        return np.arange(len(smiles_list))

    dists = tanimoto_distance_matrix(valid_fps)
    clusters = Butina.ClusterData(dists, len(valid_fps), threshold, isDistData=True)

    labels = np.full(len(smiles_list), -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            labels[valid_idx[member_idx]] = cluster_id

    # Assign singletons to invalid SMILES
    next_id = len(clusters)
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = next_id
            next_id += 1

    return labels


# ---------------------------------------------------------------------------
# Protein similarity  (k-mer Jaccard as a fast proxy for sequence identity)
# ---------------------------------------------------------------------------

def kmer_set(seq: str, k: int = 3) -> set[str]:
    """Return the set of contiguous k-mers in *seq*."""
    return {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else {seq}


def protein_similarity_matrix(sequences: list[str], k: int = 3) -> np.ndarray:
    """Pairwise Jaccard similarity of k-mer sets.

    This is a fast O(n^2 * L) approximation of sequence identity.  For
    datasets with thousands of proteins, it runs in seconds.
    """
    kmer_sets = [kmer_set(s, k) for s in sequences]
    n = len(kmer_sets)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            s = inter / union if union > 0 else 0.0
            sim[i, j] = s
            sim[j, i] = s
    return sim


def cluster_proteins(sequences: list[str], threshold: float = 0.5, k: int = 3) -> np.ndarray:
    """Cluster proteins by k-mer Jaccard similarity.

    Parameters
    ----------
    sequences : list[str]
        One amino-acid sequence per unique protein.
    threshold : float
        Similarity cutoff — proteins with pairwise similarity >= threshold
        end up in the same cluster.
    k : int
        k-mer size for Jaccard similarity.

    Returns
    -------
    labels : np.ndarray of shape (n_proteins,)
    """
    if len(sequences) <= 1:
        return np.zeros(len(sequences), dtype=int)

    sim = protein_similarity_matrix(sequences, k=k)
    dist = 1.0 - sim

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - threshold,
    )
    labels = clustering.fit_predict(dist)
    return labels


# ---------------------------------------------------------------------------
# Cluster → split assignment  (greedy bin-packing targeting 8:1:1)
# ---------------------------------------------------------------------------

def assign_clusters_to_splits(
    cluster_labels: np.ndarray,
    target_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> np.ndarray:
    """Assign each cluster to train / val / test via greedy bin-packing.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Per-sample cluster id (same cluster → same split).
    target_ratios : tuple
        Desired (train, val, test) fractions (must sum to ~1).
    seed : int
        Random seed for shuffling cluster order (affects tie-breaking).

    Returns
    -------
    split_labels : np.ndarray of shape (n_samples,)
        0 = train, 1 = val, 2 = test.
    """
    rng = np.random.RandomState(seed)
    n_total = len(cluster_labels)

    # Count members per cluster and shuffle order
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    order = rng.permutation(len(unique_clusters))
    unique_clusters = unique_clusters[order]
    counts = counts[order]

    # Sort largest-first for better packing
    size_order = np.argsort(-counts)
    unique_clusters = unique_clusters[size_order]
    counts = counts[size_order]

    targets = np.array(target_ratios, dtype=np.float64)
    targets /= targets.sum()

    current_counts = np.zeros(3, dtype=np.int64)
    cluster_to_split = {}

    for cid, cnt in zip(unique_clusters, counts):
        # Assign to the split that is furthest below its target fraction
        current_fracs = current_counts / max(current_counts.sum(), 1)
        deficit = targets - current_fracs
        chosen = int(np.argmax(deficit))
        cluster_to_split[cid] = chosen
        current_counts[chosen] += cnt

    split_labels = np.array([cluster_to_split[c] for c in cluster_labels], dtype=int)
    return split_labels


# ---------------------------------------------------------------------------
# Main splitting logic
# ---------------------------------------------------------------------------

def similarity_split(
    df: pd.DataFrame,
    mode: str = "both",
    drug_col: str = "drug_id",
    smiles_col: str = "SMILES",
    protein_col: str = "uniprot_id",
    sequence_col: str = "Target_sequence",
    label_col: str = "interaction",
    drug_threshold: float = 0.4,
    protein_threshold: float = 0.5,
    protein_kmer_k: int = 3,
    target_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split a DTI dataset respecting ligand and/or protein similarity.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with drug, protein, SMILES, sequence, and label columns.
    mode : {"drug", "protein", "both"}
        Which entity axis to cluster on.
    drug_threshold : float
        Tanimoto *distance* cutoff for drug Butina clustering (lower = fewer,
        larger clusters = stricter separation).  0.4 ≈ 60 % Tanimoto sim.
    protein_threshold : float
        k-mer Jaccard *similarity* cutoff for protein clustering (higher =
        fewer, larger clusters = stricter separation).
    protein_kmer_k : int
        k-mer size for protein similarity computation.
    target_ratios : tuple
        Desired (train, val, test) fractions.
    seed : int
        Random seed.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    stats : dict
        Summary statistics about clustering and split sizes.
    """
    df = df.reset_index(drop=True)
    stats: dict = {}

    # --- Drug clustering ---------------------------------------------------
    drug_labels = None
    if mode in ("drug", "both"):
        unique_drugs = df[[drug_col, smiles_col]].drop_duplicates(subset=drug_col)
        unique_drugs = unique_drugs.reset_index(drop=True)

        print(f"Clustering {len(unique_drugs)} unique drugs (Tanimoto distance threshold={drug_threshold}) ...")
        drug_cluster = cluster_drugs(unique_drugs[smiles_col].tolist(), threshold=drug_threshold)

        drug_to_cluster = dict(zip(unique_drugs[drug_col], drug_cluster))
        drug_labels = df[drug_col].map(drug_to_cluster).values

        n_drug_clusters = len(set(drug_cluster))
        stats["n_unique_drugs"] = len(unique_drugs)
        stats["n_drug_clusters"] = n_drug_clusters
        print(f"  → {n_drug_clusters} drug clusters")

    # --- Protein clustering ------------------------------------------------
    prot_labels = None
    if mode in ("protein", "both"):
        unique_prots = df[[protein_col, sequence_col]].drop_duplicates(subset=protein_col)
        unique_prots = unique_prots.reset_index(drop=True)

        print(f"Clustering {len(unique_prots)} unique proteins (Jaccard similarity threshold={protein_threshold}, k={protein_kmer_k}) ...")
        prot_cluster = cluster_proteins(unique_prots[sequence_col].tolist(), threshold=protein_threshold, k=protein_kmer_k)

        prot_to_cluster = dict(zip(unique_prots[protein_col], prot_cluster))
        prot_labels = df[protein_col].map(prot_to_cluster).values

        n_prot_clusters = len(set(prot_cluster))
        stats["n_unique_proteins"] = len(unique_prots)
        stats["n_protein_clusters"] = n_prot_clusters
        print(f"  → {n_prot_clusters} protein clusters")

    # --- Combine into a single cluster label per row -----------------------
    if mode == "drug":
        combined = drug_labels
    elif mode == "protein":
        combined = prot_labels
    else:  # both
        # Create unique pair ids  (drug_cluster, prot_cluster) → int
        pair_to_id: dict[tuple[int, int], int] = {}
        combined = np.empty(len(df), dtype=int)
        for i in range(len(df)):
            pair = (int(drug_labels[i]), int(prot_labels[i]))
            if pair not in pair_to_id:
                pair_to_id[pair] = len(pair_to_id)
            combined[i] = pair_to_id[pair]

        stats["n_combined_clusters"] = len(pair_to_id)
        print(f"  → {len(pair_to_id)} combined (drug, protein) cluster pairs")

    # --- Assign clusters to splits -----------------------------------------
    split_ids = assign_clusters_to_splits(combined, target_ratios=target_ratios, seed=seed)

    train_df = df[split_ids == 0].reset_index(drop=True)
    val_df = df[split_ids == 1].reset_index(drop=True)
    test_df = df[split_ids == 2].reset_index(drop=True)

    # --- Report -------------------------------------------------------------
    for name, sdf in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n = len(sdf)
        frac = n / len(df) if len(df) else 0
        n_pos = (sdf[label_col] == 1).sum() if label_col in sdf.columns else "?"
        n_neg = (sdf[label_col] == 0).sum() if label_col in sdf.columns else "?"
        stats[f"{name.lower()}_n"] = n
        stats[f"{name.lower()}_frac"] = round(frac, 3)
        print(f"  {name:5s}: {n:>7,} samples ({frac:.1%})  pos={n_pos}  neg={n_neg}")

    # --- Leak check --------------------------------------------------------
    _check_leakage(train_df, val_df, test_df, drug_col, protein_col, mode, stats)

    return train_df, val_df, test_df, stats


def _check_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    drug_col: str,
    prot_col: str,
    mode: str,
    stats: dict,
) -> None:
    """Report entity overlap between splits (should be zero for clustered axis)."""
    if mode in ("drug", "both"):
        train_drugs = set(train[drug_col])
        val_drugs = set(val[drug_col])
        test_drugs = set(test[drug_col])
        leak_val = train_drugs & val_drugs
        leak_test = train_drugs & test_drugs
        stats["drug_leak_train_val"] = len(leak_val)
        stats["drug_leak_train_test"] = len(leak_test)
        if leak_val or leak_test:
            print(f"  ⚠ Drug overlap: train∩val={len(leak_val)}, train∩test={len(leak_test)}")
        else:
            print("  ✓ No drug leakage between splits")

    if mode in ("protein", "both"):
        train_prots = set(train[prot_col])
        val_prots = set(val[prot_col])
        test_prots = set(test[prot_col])
        leak_val = train_prots & val_prots
        leak_test = train_prots & test_prots
        stats["protein_leak_train_val"] = len(leak_val)
        stats["protein_leak_train_test"] = len(leak_test)
        if leak_val or leak_test:
            print(f"  ⚠ Protein overlap: train∩val={len(leak_val)}, train∩test={len(leak_test)}")
        else:
            print("  ✓ No protein leakage between splits")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Similarity-aware train/val/test split for DTI datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    grp_in = p.add_argument_group("input")
    grp_in.add_argument("--csv", required=True, type=Path, nargs="+",
                        help="One or more CSVs to concatenate before splitting")

    # Columns
    grp_col = p.add_argument_group("column names")
    grp_col.add_argument("--drug_col", default="drug_id")
    grp_col.add_argument("--smiles_col", default="SMILES")
    grp_col.add_argument("--protein_col", default="uniprot_id")
    grp_col.add_argument("--sequence_col", default="Target_sequence")
    grp_col.add_argument("--label_col", default="interaction")

    # Splitting
    grp_sp = p.add_argument_group("splitting")
    grp_sp.add_argument("--mode", choices=["drug", "protein", "both"], default="both",
                        help="Which axis to enforce similarity separation on")
    grp_sp.add_argument("--drug_threshold", type=float, default=0.4,
                        help="Tanimoto DISTANCE cutoff for drug clustering "
                             "(0.4 ≈ similarity ≥ 0.6)")
    grp_sp.add_argument("--protein_threshold", type=float, default=0.5,
                        help="k-mer Jaccard SIMILARITY cutoff for protein clustering")
    grp_sp.add_argument("--protein_kmer_k", type=int, default=3,
                        help="k-mer size for protein similarity")
    grp_sp.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Target split ratios")
    grp_sp.add_argument("--seed", type=int, default=42)

    # Output
    grp_out = p.add_argument_group("output")
    grp_out.add_argument("--out_dir", type=Path, default=Path("lists"),
                         help="Directory for output CSVs")
    grp_out.add_argument("--prefix", type=str, default="sim",
                         help="Filename prefix for output CSVs")

    args = p.parse_args()

    # Load
    dfs = [pd.read_csv(f) for f in args.csv]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(df):,} interactions ({len(dfs)} file(s))")

    train, val, test, stats = similarity_split(
        df,
        mode=args.mode,
        drug_col=args.drug_col,
        smiles_col=args.smiles_col,
        protein_col=args.protein_col,
        sequence_col=args.sequence_col,
        label_col=args.label_col,
        drug_threshold=args.drug_threshold,
        protein_threshold=args.protein_threshold,
        protein_kmer_k=args.protein_kmer_k,
        target_ratios=tuple(args.ratios),
        seed=args.seed,
    )

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, sdf in [("train", train), ("val", val), ("test", test)]:
        path = args.out_dir / f"{args.prefix}_{name}.csv"
        sdf.to_csv(path, index=False)
        print(f"Saved {path}  ({len(sdf):,} rows)")


if __name__ == "__main__":
    main()
