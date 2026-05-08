"""Shared utilities for the Clever Hans tests (test-only ablation and full retrain).

Per-dimension train-marginal statistics for drug and protein embeddings, weighted
by pair frequency in the train CSV. Used to draw matched-marginal Gaussian noise
that replaces one side at train and/or test time.
"""

from __future__ import annotations

import glob
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _per_dim_stats_drug(
    train_csv: Path,
    drug_dir: str,
    allow_unimol_suffix: bool,
    trim_terminal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(train_csv, usecols=["drug_id"], dtype={"drug_id": str})
    counts = Counter(df["drug_id"].astype(str).tolist())

    sum_v: torch.Tensor | None = None
    sum_sq: torch.Tensor | None = None
    n_tokens = 0
    n_missing = 0

    for drug_id, c in counts.items():
        path = None
        if allow_unimol_suffix:
            cand = os.path.join(drug_dir, f"{drug_id}_unimol.pt")
            if os.path.exists(cand):
                path = cand
        if path is None:
            cand = os.path.join(drug_dir, f"{drug_id}.pt")
            if os.path.exists(cand):
                path = cand
        if path is None:
            n_missing += 1
            continue

        d = torch.load(path, map_location="cpu", weights_only=False)
        arr = np.asarray(d["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
        emb = torch.from_numpy(arr)
        if trim_terminal and emb.shape[0] > 1:
            emb = emb[:-1, :]

        emb_d = emb.double()
        s = emb_d.sum(dim=0) * c
        ss = (emb_d ** 2).sum(dim=0) * c
        if sum_v is None:
            sum_v = s.clone()
            sum_sq = ss.clone()
        else:
            sum_v += s
            sum_sq += ss
        n_tokens += c * emb.shape[0]

    if n_missing:
        print(f"  [drug] {n_missing} ids in train CSV missing on disk (skipped)")
    if sum_v is None or n_tokens == 0:
        raise RuntimeError("No drug embeddings found for stats")

    mean = (sum_v / n_tokens).float()
    var = (sum_sq / n_tokens).float() - mean ** 2
    std = var.clamp(min=1e-8).sqrt()
    return mean, std


def _per_dim_stats_protein(
    train_csv: Path,
    protein_dir: str,
    trim_terminal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(train_csv, usecols=["uniprot_id"], dtype={"uniprot_id": str})
    counts = Counter(df["uniprot_id"].astype(str).tolist())

    sum_v: torch.Tensor | None = None
    sum_sq: torch.Tensor | None = None
    n_tokens = 0
    n_missing = 0

    for uid, c in counts.items():
        files = sorted(glob.glob(os.path.join(protein_dir, uid, "*.pt")))
        if not files:
            n_missing += 1
            continue
        emb = torch.load(files[-1], map_location="cpu", weights_only=True)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.to(dtype=torch.float32)
        if trim_terminal and emb.shape[0] > 1:
            emb = emb[:-1, :]

        emb_d = emb.double()
        s = emb_d.sum(dim=0) * c
        ss = (emb_d ** 2).sum(dim=0) * c
        if sum_v is None:
            sum_v = s.clone()
            sum_sq = ss.clone()
        else:
            sum_v += s
            sum_sq += ss
        n_tokens += c * emb.shape[0]

    if n_missing:
        print(f"  [protein] {n_missing} ids in train CSV missing on disk (skipped)")
    if sum_v is None or n_tokens == 0:
        raise RuntimeError("No protein embeddings found for stats")

    mean = (sum_v / n_tokens).float()
    var = (sum_sq / n_tokens).float() - mean ** 2
    std = var.clamp(min=1e-8).sqrt()
    return mean, std


def get_or_compute_stats(
    train_csv: str,
    protein_dir: str,
    drug_dir: str,
    *,
    tag: str,
    cache_dir: Path,
    trim_terminal: bool,
    allow_unimol_suffix: bool,
    verbose: bool = True,
) -> dict:
    """Return dict with drug_mean/drug_std/prot_mean/prot_std (all 1-D float tensors)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    drug_path = cache_dir / f"{tag}_drug.pt"
    prot_path = cache_dir / f"{tag}_protein.pt"

    if drug_path.exists():
        d = torch.load(drug_path, weights_only=True)
        drug_mean, drug_std = d["mean"], d["std"]
        if verbose:
            print(f"  [{tag}] loaded drug stats  D={drug_mean.numel()}")
    else:
        if verbose:
            print(f"  [{tag}] computing drug stats from {train_csv}...")
        drug_mean, drug_std = _per_dim_stats_drug(
            Path(train_csv), drug_dir, allow_unimol_suffix, trim_terminal
        )
        torch.save({"mean": drug_mean, "std": drug_std}, drug_path)
        if verbose:
            print(f"  [{tag}] saved {drug_path}  D={drug_mean.numel()}")

    if prot_path.exists():
        d = torch.load(prot_path, weights_only=True)
        prot_mean, prot_std = d["mean"], d["std"]
        if verbose:
            print(f"  [{tag}] loaded protein stats  D={prot_mean.numel()}")
    else:
        if verbose:
            print(f"  [{tag}] computing protein stats from {train_csv}...")
        prot_mean, prot_std = _per_dim_stats_protein(
            Path(train_csv), protein_dir, trim_terminal
        )
        torch.save({"mean": prot_mean, "std": prot_std}, prot_path)
        if verbose:
            print(f"  [{tag}] saved {prot_path}  D={prot_mean.numel()}")

    return {
        "drug_mean": drug_mean, "drug_std": drug_std,
        "prot_mean": prot_mean, "prot_std": prot_std,
    }
