"""Clever Hans test-only ablation for DTI models.

For a trained checkpoint, evaluate the test split under three conditions and
compare metrics:
  1. baseline       — real drug + real protein embeddings
  2. drug_noise     — drug emb replaced with Gaussian noise matching the per-dim
                      marginal stats of the TRAIN split's drug embeddings
  3. protein_noise  — protein emb replaced with Gaussian noise matching the
                      per-dim marginal stats of the TRAIN split's protein embs

Stats are weighted by pair frequency in the train CSV (not unique-entity uniform),
so they reflect the marginal the model actually trained on. Noise is per-token
(randn_like * std + mean) and padded positions are zeroed back to match the
collate convention. Multiple draws are averaged to give mean ± std.

Usage
-----
    python clever_hans_test.py \
        --ckpt saved/model_2415696658070472781_epoch_79.pt \
        --tag drug_cs \
        --n_draws 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config.cfg import get_cfg_defaults
from dataset import MyDataset, collate_fn
from model import Model


def _per_dim_stats_drug(
    train_csv: Path,
    drug_dir: str,
    allow_unimol_suffix: bool,
    trim_terminal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream train CSV, weight by pair frequency, return per-dim (mean, std)."""
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


def get_or_compute_stats(cfg, tag: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    drug_path = cache_dir / f"{tag}_drug.pt"
    prot_path = cache_dir / f"{tag}_protein.pt"

    probe = MyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    trim = probe.trim_terminal_token
    allow_uni = probe.allow_unimol_suffix
    print(f"  dataset_kind={probe.dataset_kind}  trim_terminal={trim}  allow_unimol_suffix={allow_uni}")
    del probe

    if drug_path.exists():
        d = torch.load(drug_path, weights_only=True)
        drug_mean, drug_std = d["mean"], d["std"]
        print(f"  loaded drug stats from {drug_path}  D={drug_mean.numel()}")
    else:
        print(f"  computing drug stats from {cfg.DATA.TRAIN_CSV_PATH}...")
        drug_mean, drug_std = _per_dim_stats_drug(
            Path(cfg.DATA.TRAIN_CSV_PATH), cfg.DATA.DRUG_DIR, allow_uni, trim
        )
        torch.save({"mean": drug_mean, "std": drug_std}, drug_path)
        print(f"  saved {drug_path}  D={drug_mean.numel()}")

    if prot_path.exists():
        d = torch.load(prot_path, weights_only=True)
        prot_mean, prot_std = d["mean"], d["std"]
        print(f"  loaded protein stats from {prot_path}  D={prot_mean.numel()}")
    else:
        print(f"  computing protein stats from {cfg.DATA.TRAIN_CSV_PATH}...")
        prot_mean, prot_std = _per_dim_stats_protein(
            Path(cfg.DATA.TRAIN_CSV_PATH), cfg.DATA.PROTEIN_DIR, trim
        )
        torch.save({"mean": prot_mean, "std": prot_std}, prot_path)
        print(f"  saved {prot_path}  D={prot_mean.numel()}")

    print(
        "  drug    mean range: [{:.4f}, {:.4f}]  std range: [{:.4f}, {:.4f}]".format(
            drug_mean.min().item(), drug_mean.max().item(),
            drug_std.min().item(), drug_std.max().item(),
        )
    )
    print(
        "  protein mean range: [{:.4f}, {:.4f}]  std range: [{:.4f}, {:.4f}]".format(
            prot_mean.min().item(), prot_mean.max().item(),
            prot_std.min().item(), prot_std.max().item(),
        )
    )
    return {
        "drug_mean": drug_mean, "drug_std": drug_std,
        "prot_mean": prot_mean, "prot_std": prot_std,
    }


def _noise_like(real: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """real: (B, L, D); mask: (B, L) True=padding. Padded positions are zeroed."""
    m = mean.view(1, 1, -1).to(device=real.device, dtype=real.dtype)
    s = std.view(1, 1, -1).to(device=real.device, dtype=real.dtype)
    noise = torch.randn_like(real) * s + m
    noise = noise.masked_fill(mask.unsqueeze(-1), 0.0)
    return noise


@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
    *,
    mode: str,
    stats: dict | None = None,
    seed: int | None = None,
) -> dict:
    """mode: 'baseline' | 'drug_noise' | 'protein_noise'."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    non_blocking = device.type == "cuda"

    for batch in loader:
        labels = batch["label"].to(device, non_blocking=non_blocking)
        protein_mask = batch["protein_mask"].to(device, non_blocking=non_blocking)
        drug_mask = batch["drug_mask"].to(device, non_blocking=non_blocking)
        protein_emb = batch["protein_emb"].to(device, non_blocking=non_blocking)
        drug_emb = batch["drug_emb"].to(device, non_blocking=non_blocking)

        if mode == "drug_noise":
            drug_emb = _noise_like(drug_emb, stats["drug_mean"], stats["drug_std"], drug_mask)
        elif mode == "protein_noise":
            protein_emb = _noise_like(protein_emb, stats["prot_mean"], stats["prot_std"], protein_mask)
        elif mode != "baseline":
            raise ValueError(f"unknown mode: {mode}")

        logits = model(
            protein_emb, drug_emb,
            protein_mask=protein_mask, drug_mask=drug_mask,
            mode="test",
        )
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        labels_np = labels.detach().cpu().numpy().astype(int)
        preds_all.append(preds)
        labels_all.append(labels_np)
        probs_all.append(probs)

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return {
            "ACC":   float(accuracy_score(labels, preds) * 100),
            "PPV":   float(precision_score(labels, preds, zero_division=0)),
            "TPR":   float(recall_score(labels, preds, zero_division=0)),
            "F1":    float(f1_score(labels, preds, zero_division=0)),
            "MCC":   float(matthews_corrcoef(labels, preds)),
            "AUROC": float(roc_auc_score(labels, probs)),
            "AUPRC": float(average_precision_score(labels, probs)),
        }


def _summarize(draws: list[dict]) -> dict:
    if not draws:
        return {}
    keys = list(draws[0].keys())
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in draws], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clever Hans test-only ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tag", type=str, required=True,
                   help="dataset identifier; used as cache key for stats files")
    p.add_argument("--n_draws", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=Path, default=Path("stats/clever_hans"))
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 keeps the dataset's lru_cache enabled (fastest on macOS); "
                        ">0 forces caches off to allow pickling to workers")
    args = p.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    cfg = get_cfg_defaults()
    print(f"Train CSV (for stats): {cfg.DATA.TRAIN_CSV_PATH}")
    print(f"Test  CSV (for eval):  {cfg.DATA.TEST_CSV_PATH}")

    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Model(cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nResolving train-marginal stats:")
    stats = get_or_compute_stats(cfg, args.tag, args.cache_dir)
    for k in ("drug_mean", "drug_std", "prot_mean", "prot_std"):
        stats[k] = stats[k].to(device)

    # lru_cache-wrapped methods can't be pickled to spawned workers on macOS;
    # disable the cache when running with workers, keep it on for the in-process path.
    use_cache = args.num_workers == 0
    test_ds = MyDataset(
        cfg.DATA.TEST_CSV_PATH,
        cfg.DATA.PROTEIN_DIR,
        cfg.DATA.DRUG_DIR,
        prot_cache_size=4096 if use_cache else 0,
        drug_cache_size=4096 if use_cache else 0,
    )
    loader_kwargs = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    test_dl = DataLoader(test_ds, **loader_kwargs)

    print("\n=== Baseline (real / real) ===")
    base = run_inference(model, test_dl, device, mode="baseline")
    for k, v in base.items():
        print(f"  {k}: {v:.6f}")

    drug_draws: list[dict] = []
    prot_draws: list[dict] = []

    for draw in range(args.n_draws):
        s = args.seed + draw
        print(f"\n=== Draw {draw + 1}/{args.n_draws}  seed={s} ===")

        print("  drug-noised:")
        r_d = run_inference(model, test_dl, device, mode="drug_noise", stats=stats, seed=s)
        for k, v in r_d.items():
            print(f"    {k}: {v:.6f}")
        drug_draws.append(r_d)

        print("  protein-noised:")
        r_p = run_inference(model, test_dl, device, mode="protein_noise", stats=stats, seed=s + 10_000)
        for k, v in r_p.items():
            print(f"    {k}: {v:.6f}")
        prot_draws.append(r_p)

    summary = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "train_csv": str(cfg.DATA.TRAIN_CSV_PATH),
        "test_csv": str(cfg.DATA.TEST_CSV_PATH),
        "n_draws": args.n_draws,
        "seed": args.seed,
        "baseline": base,
        "drug_noise_summary": _summarize(drug_draws),
        "protein_noise_summary": _summarize(prot_draws),
        "drug_noise_draws": drug_draws,
        "protein_noise_draws": prot_draws,
    }

    print("\n=== Summary (baseline vs noised mean over draws) ===")
    print(f"  metric  | baseline   | drug-noise (Δ)         | protein-noise (Δ)")
    for k in base.keys():
        b = base[k]
        d = summary["drug_noise_summary"][k]["mean"]
        ds = summary["drug_noise_summary"][k]["std"]
        p = summary["protein_noise_summary"][k]["mean"]
        ps = summary["protein_noise_summary"][k]["std"]
        print(f"  {k:6s}  | {b:9.5f}  | {d:9.5f} ± {ds:7.5f} (Δ={d - b:+.5f})  | {p:9.5f} ± {ps:7.5f} (Δ={p - b:+.5f})")

    out_json = args.out_json or (args.cache_dir / f"clever_hans_{args.tag}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
