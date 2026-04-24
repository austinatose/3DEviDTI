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
import tempfile
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


def cluster_proteins_mmseqs2(sequences: list[str], threshold: float = 0.5) -> np.ndarray:
    """Cluster proteins with MMseqs2 via the pymmseqs wrapper.

    Parameters
    ----------
    sequences : list[str]
        One amino-acid sequence per unique protein.
    threshold : float
        MMseqs2 minimum sequence identity (`min_seq_id`).

    Returns
    -------
    labels : np.ndarray of shape (n_proteins,)
    """
    if len(sequences) <= 1:
        return np.zeros(len(sequences), dtype=int)

    try:
        from pymmseqs.commands import easy_cluster
    except Exception as e:
        raise ImportError(
            "pymmseqs is required for protein MMseqs2 clustering. "
            "Install it with: pip install pymmseqs"
        ) from e

    with tempfile.TemporaryDirectory(prefix="pymmseqs_split_") as tmpdir:
        tmp_path = Path(tmpdir)
        fasta_path = tmp_path / "proteins.fasta"
        out_prefix = tmp_path / "prot_cluster"
        mmseqs_tmp = tmp_path / "mmseqs_tmp"
        mmseqs_tmp.mkdir(parents=True, exist_ok=True)

        with fasta_path.open("w", encoding="utf-8") as handle:
            for i, seq in enumerate(sequences):
                # IDs are simple and deterministic so we can map clusters back.
                handle.write(f">seq_{i}\n{str(seq)}\n")

        clustering_result = easy_cluster(
            str(fasta_path),
            str(out_prefix),
            str(mmseqs_tmp),
            min_seq_id=float(threshold),
        )

        id_to_index = {f"seq_{i}": i for i in range(len(sequences))}
        labels = np.full(len(sequences), -1, dtype=int)

        for cluster_id, cluster in enumerate(clustering_result.to_gen()):
            ids: set[str] = set()

            if isinstance(cluster, dict):
                rep = cluster.get("rep")
                if rep is not None:
                    ids.add(str(rep))
                for member in cluster.get("members", []):
                    ids.add(str(member))
            elif isinstance(cluster, (tuple, list)) and len(cluster) >= 2:
                ids.add(str(cluster[0]))
                members = cluster[1] if isinstance(cluster[1], (tuple, list, set)) else [cluster[1]]
                for member in members:
                    ids.add(str(member))

            for seq_id in ids:
                idx = id_to_index.get(seq_id)
                if idx is not None:
                    labels[idx] = cluster_id

        # Any unassigned sequence becomes its own singleton cluster.
        next_cluster_id = int(labels.max()) + 1 if (labels >= 0).any() else 0
        for i in range(len(labels)):
            if labels[i] < 0:
                labels[i] = next_cluster_id
                next_cluster_id += 1

    return labels


def drug_similarity_matrix(smiles_list: list[str]) -> np.ndarray:
    """Pairwise Tanimoto similarity matrix for SMILES strings."""
    try:
        from rdkit import DataStructs
    except Exception as e:
        raise ImportError(
            "RDKit is required for drug similarity computations. "
            "Install rdkit or use protein-only mode."
        ) from e

    fps = ecfp4_fingerprints(smiles_list)
    n = len(fps)
    sim = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        sim[i, i] = 1.0
        if fps[i] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim[i, :] = np.asarray(sims, dtype=np.float32)

    # Enforce symmetry and valid range.
    sim = np.maximum(sim, sim.T)
    np.clip(sim, 0.0, 1.0, out=sim)
    return sim


def cluster_distance_from_entity_similarity(
    entity_cluster_labels: np.ndarray,
    sim_matrix: np.ndarray,
    agg: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cluster-to-cluster distance from entity similarity matrix.

    `agg=max` yields `1 - max(sim)` and `agg=mean` yields `1 - mean(sim)`.
    """
    unique_clusters = np.unique(entity_cluster_labels)
    cluster_to_indices = {
        c: np.where(entity_cluster_labels == c)[0]
        for c in unique_clusters
    }

    n = len(unique_clusters)
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        idx_i = cluster_to_indices[unique_clusters[i]]
        for j in range(i + 1, n):
            idx_j = cluster_to_indices[unique_clusters[j]]
            block = sim_matrix[np.ix_(idx_i, idx_j)]
            if block.size == 0:
                sim_val = 0.0
            elif agg == "max":
                sim_val = float(np.max(block))
            elif agg == "mean":
                sim_val = float(np.mean(block))
            else:
                raise ValueError(f"Unknown cluster distance aggregator: {agg!r}")

            d = 1.0 - sim_val
            dist[i, j] = d
            dist[j, i] = d

    return unique_clusters, dist


def assignment_objective(assign: np.ndarray, sizes: np.ndarray, dist: np.ndarray, targets: np.ndarray) -> float:
    """Higher is better: larger inter-split distance with ratio adherence."""
    inter_weighted = 0.0
    inter_wsum = 0.0
    n = len(assign)
    for i in range(n):
        for j in range(i + 1, n):
            if assign[i] == assign[j]:
                continue
            w = float(sizes[i] * sizes[j])
            inter_weighted += w * float(dist[i, j])
            inter_wsum += w

    inter_score = inter_weighted / inter_wsum if inter_wsum > 0 else 0.0

    split_counts = np.array([
        int(np.sum(sizes[assign == 0])),
        int(np.sum(sizes[assign == 1])),
        int(np.sum(sizes[assign == 2])),
    ], dtype=np.float64)
    size_penalty = np.sum(np.abs(split_counts - targets) / np.maximum(targets, 1.0))
    return inter_score - 0.15 * float(size_penalty)


def assign_clusters_distance_maximization(
    row_cluster_labels: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_dist: np.ndarray,
    target_ratios: tuple[float, float, float],
    seed: int = 42,
    refine_iters: int = 0,
    size_tolerance: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Distance-aware cluster assignment for train/val/test.

    Procedure:
    1) Build a representative train core first.
    2) Assign remaining clusters to val/test in decreasing distance-to-train.
    3) Optionally refine by local cluster swaps.
    """
    rng = np.random.RandomState(seed)

    cluster_ids = np.asarray(cluster_ids)
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return np.zeros(len(row_cluster_labels), dtype=int), {"distance_refine_swaps": 0}

    uniq_rows, row_counts = np.unique(row_cluster_labels, return_counts=True)
    count_map = dict(zip(uniq_rows.tolist(), row_counts.tolist()))
    sizes = np.array([int(count_map.get(c, 0)) for c in cluster_ids], dtype=np.int64)

    n_rows = len(row_cluster_labels)
    targets = np.array(target_ratios, dtype=np.float64)
    targets = targets / max(np.sum(targets), 1e-12)
    target_counts = targets * n_rows

    assign = np.full(n_clusters, -1, dtype=np.int8)
    unassigned = set(range(n_clusters))
    split_counts = np.zeros(3, dtype=np.int64)

    # Step 1: representative training core.
    avg_dist = np.mean(cluster_dist, axis=1)
    seed_idx = int(np.argmin(avg_dist))
    assign[seed_idx] = 0
    unassigned.remove(seed_idx)
    split_counts[0] += sizes[seed_idx]

    while unassigned and split_counts[0] < target_counts[0]:
        train_idx = np.where(assign == 0)[0]
        best_idx = None
        best_score = -np.inf
        for idx in unassigned:
            rep_score = -float(np.mean(cluster_dist[idx, train_idx]))
            overshoot = max(
                0.0,
                (split_counts[0] + sizes[idx] - target_counts[0]) / max(target_counts[0], 1.0),
            )
            score = rep_score - 0.25 * overshoot + 1e-9 * rng.randn()
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        assign[best_idx] = 0
        unassigned.remove(best_idx)
        split_counts[0] += sizes[best_idx]

    # Step 2: farthest-from-train assignment into val/test.
    while unassigned:
        train_idx = np.where(assign == 0)[0]
        cand = max(
            unassigned,
            key=lambda i: (float(np.min(cluster_dist[i, train_idx])), int(sizes[i])),
        )

        val_def = target_counts[1] - split_counts[1]
        test_def = target_counts[2] - split_counts[2]

        if val_def <= 0 < test_def:
            chosen = 2
        elif test_def <= 0 < val_def:
            chosen = 1
        else:
            val_need = val_def / max(target_counts[1], 1.0)
            test_need = test_def / max(target_counts[2], 1.0)
            if val_need > test_need:
                chosen = 1
            elif test_need > val_need:
                chosen = 2
            else:
                chosen = 1 if split_counts[1] <= split_counts[2] else 2

        assign[cand] = chosen
        unassigned.remove(cand)
        split_counts[chosen] += sizes[cand]

    # Step 3: optional local refinement via swap moves.
    swaps = 0
    if refine_iters > 0 and n_clusters > 1:
        tol_rows = np.maximum(target_counts * size_tolerance, float(np.max(sizes)))
        current_obj = assignment_objective(assign, sizes, cluster_dist, target_counts)

        for _ in range(refine_iters):
            best_pair = None
            best_obj = current_obj

            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    if assign[i] == assign[j]:
                        continue

                    s_i = int(assign[i])
                    s_j = int(assign[j])
                    new_counts = split_counts.astype(np.float64).copy()
                    new_counts[s_i] += sizes[j] - sizes[i]
                    new_counts[s_j] += sizes[i] - sizes[j]

                    if np.any(np.abs(new_counts - target_counts) > tol_rows):
                        continue

                    trial = assign.copy()
                    trial[i], trial[j] = trial[j], trial[i]
                    obj = assignment_objective(trial, sizes, cluster_dist, target_counts)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            s_i = int(assign[i])
            s_j = int(assign[j])
            split_counts[s_i] += sizes[j] - sizes[i]
            split_counts[s_j] += sizes[i] - sizes[j]
            assign[i], assign[j] = assign[j], assign[i]
            current_obj = best_obj
            swaps += 1

    cluster_to_split = {int(c): int(assign[i]) for i, c in enumerate(cluster_ids)}
    split_ids = np.array([cluster_to_split[int(c)] for c in row_cluster_labels], dtype=int)

    stats = {
        "distance_refine_swaps": int(swaps),
        "distance_objective": float(assignment_objective(assign, sizes, cluster_dist, target_counts)),
    }
    return split_ids, stats


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
    protein_cluster_method: str = "jaccard",
    split_assignment: str = "greedy",
    cluster_distance_agg: str = "max",
    distance_refine_iters: int = 0,
    distance_size_tolerance: float = 0.02,
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
        Protein clustering threshold. For jaccard, this is the k-mer Jaccard
        similarity cutoff. For mmseqs2, this is MMseqs2 `min_seq_id`.
    protein_kmer_k : int
        k-mer size for jaccard protein similarity computation.
    protein_cluster_method : {"jaccard", "mmseqs2"}
        Backend used for protein clustering.
    split_assignment : {"greedy", "distance_max"}
        Cluster-to-split assignment strategy.
    cluster_distance_agg : {"max", "mean"}
        Cluster distance aggregation rule used by distance-maximization.
    distance_refine_iters : int
        Number of local swap refinement passes for distance-maximization.
    distance_size_tolerance : float
        Allowed relative split-size deviation during refinement swaps.
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
    stats["split_assignment"] = split_assignment

    # --- Drug clustering ---------------------------------------------------
    drug_labels = None
    drug_cluster_ids = None
    drug_cluster_dist = None
    if mode in ("drug", "both"):
        unique_drugs = df[[drug_col, smiles_col]].groupby(drug_col, as_index=False).first()

        print(f"Clustering {len(unique_drugs)} unique drugs (Tanimoto distance threshold={drug_threshold}) ...")
        drug_cluster = cluster_drugs(unique_drugs[smiles_col].tolist(), threshold=drug_threshold)

        drug_to_cluster = dict(zip(unique_drugs[drug_col], drug_cluster))
        drug_labels = df[drug_col].map(drug_to_cluster).values

        n_drug_clusters = len(set(drug_cluster))
        stats["n_unique_drugs"] = len(unique_drugs)
        stats["n_drug_clusters"] = n_drug_clusters
        print(f"  → {n_drug_clusters} drug clusters")

        if split_assignment == "distance_max":
            drug_sim = drug_similarity_matrix(unique_drugs[smiles_col].tolist())
            drug_cluster_ids, drug_cluster_dist = cluster_distance_from_entity_similarity(
                drug_cluster,
                drug_sim,
                agg=cluster_distance_agg,
            )

    # --- Protein clustering ------------------------------------------------
    prot_labels = None
    prot_cluster_ids = None
    prot_cluster_dist = None
    if mode in ("protein", "both"):
        unique_prots = df[[protein_col, sequence_col]].groupby(protein_col, as_index=False).first()

        if protein_cluster_method == "jaccard":
            print(
                f"Clustering {len(unique_prots)} unique proteins "
                f"(Jaccard similarity threshold={protein_threshold}, k={protein_kmer_k}) ..."
            )
            prot_cluster = cluster_proteins(
                unique_prots[sequence_col].tolist(),
                threshold=protein_threshold,
                k=protein_kmer_k,
            )
        elif protein_cluster_method == "mmseqs2":
            print(
                f"Clustering {len(unique_prots)} unique proteins "
                f"(MMseqs2 min_seq_id={protein_threshold}) ..."
            )
            prot_cluster = cluster_proteins_mmseqs2(
                unique_prots[sequence_col].tolist(),
                threshold=protein_threshold,
            )
        else:
            raise ValueError(
                f"Invalid protein_cluster_method={protein_cluster_method!r}. "
                "Expected one of: 'jaccard', 'mmseqs2'."
            )

        prot_to_cluster = dict(zip(unique_prots[protein_col], prot_cluster))
        prot_labels = df[protein_col].map(prot_to_cluster).values

        n_prot_clusters = len(set(prot_cluster))
        stats["n_unique_proteins"] = len(unique_prots)
        stats["n_protein_clusters"] = n_prot_clusters
        print(f"  → {n_prot_clusters} protein clusters")

        if split_assignment == "distance_max":
            if protein_cluster_method == "mmseqs2":
                print("  Note: distance-maximization with mmseqs2 uses k-mer Jaccard for cluster-distance scoring.")
            prot_sim = protein_similarity_matrix(unique_prots[sequence_col].tolist(), k=protein_kmer_k)
            prot_cluster_ids, prot_cluster_dist = cluster_distance_from_entity_similarity(
                prot_cluster,
                prot_sim,
                agg=cluster_distance_agg,
            )

    # --- Combine into a single cluster label per row -----------------------
    pair_to_id: dict[tuple[int, int], int] = {}
    if mode == "drug":
        combined = drug_labels
    elif mode == "protein":
        combined = prot_labels
    else:  # both
        if drug_labels is None or prot_labels is None:
            raise RuntimeError("Internal error: missing drug/protein cluster labels in mode='both'.")

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
    if split_assignment == "greedy":
        split_ids = assign_clusters_to_splits(combined, target_ratios=target_ratios, seed=seed)
    elif split_assignment == "distance_max":
        if mode == "drug":
            if drug_cluster_ids is None or drug_cluster_dist is None:
                raise RuntimeError("Drug cluster distances are unavailable for distance_max assignment.")
            cluster_ids = drug_cluster_ids
            cluster_dist = drug_cluster_dist
        elif mode == "protein":
            if prot_cluster_ids is None or prot_cluster_dist is None:
                raise RuntimeError("Protein cluster distances are unavailable for distance_max assignment.")
            cluster_ids = prot_cluster_ids
            cluster_dist = prot_cluster_dist
        else:
            if (
                drug_cluster_ids is None
                or drug_cluster_dist is None
                or prot_cluster_ids is None
                or prot_cluster_dist is None
            ):
                raise RuntimeError("Drug/protein cluster distances are unavailable for distance_max assignment.")

            drug_idx = {int(c): i for i, c in enumerate(drug_cluster_ids)}
            prot_idx = {int(c): i for i, c in enumerate(prot_cluster_ids)}
            id_to_pair = {v: k for k, v in pair_to_id.items()}

            cluster_ids = np.array(sorted(id_to_pair.keys()), dtype=int)
            n_comb = len(cluster_ids)
            cluster_dist = np.zeros((n_comb, n_comb), dtype=np.float32)

            for i in range(n_comb):
                dci, pci = id_to_pair[int(cluster_ids[i])]
                for j in range(i + 1, n_comb):
                    dcj, pcj = id_to_pair[int(cluster_ids[j])]
                    d_drug = float(drug_cluster_dist[drug_idx[int(dci)], drug_idx[int(dcj)]])
                    d_prot = float(prot_cluster_dist[prot_idx[int(pci)], prot_idx[int(pcj)]])
                    d = 0.5 * (d_drug + d_prot)
                    cluster_dist[i, j] = d
                    cluster_dist[j, i] = d

        split_ids, assign_stats = assign_clusters_distance_maximization(
            combined,
            cluster_ids=cluster_ids,
            cluster_dist=cluster_dist,
            target_ratios=target_ratios,
            seed=seed,
            refine_iters=distance_refine_iters,
            size_tolerance=distance_size_tolerance,
        )
        stats.update(assign_stats)
    else:
        raise ValueError(
            f"Invalid split_assignment={split_assignment!r}. "
            "Expected one of: 'greedy', 'distance_max'."
        )

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
                        help="Protein threshold: Jaccard similarity cutoff (jaccard) or min_seq_id (mmseqs2)")
    grp_sp.add_argument("--protein_kmer_k", type=int, default=3,
                        help="k-mer size for protein similarity")
    grp_sp.add_argument("--protein_cluster_method", choices=["jaccard", "mmseqs2"], default="jaccard",
                        help="Protein clustering backend")
    grp_sp.add_argument("--split_assignment", choices=["greedy", "distance_max"], default="greedy",
                        help="Cluster-to-split assignment strategy")
    grp_sp.add_argument("--cluster_distance_agg", choices=["max", "mean"], default="max",
                        help="Cluster distance rule for distance_max assignment")
    grp_sp.add_argument("--distance_refine_iters", type=int, default=0,
                        help="Local swap refinement iterations for distance_max assignment")
    grp_sp.add_argument("--distance_size_tolerance", type=float, default=0.02,
                        help="Relative split-size tolerance used during refinement swaps")
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
        protein_cluster_method=args.protein_cluster_method,
        split_assignment=args.split_assignment,
        cluster_distance_agg=args.cluster_distance_agg,
        distance_refine_iters=args.distance_refine_iters,
        distance_size_tolerance=args.distance_size_tolerance,
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
