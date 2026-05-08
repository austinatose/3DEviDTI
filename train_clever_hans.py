"""Full Clever Hans matrix: retrain with one side replaced by marginal-matched
Gaussian noise, evaluate on the (also noised) test split.

Per-token-fresh noise on the chosen side (every __getitem__ call yields a brand-
new sample). The model thus has zero per-entity information about that side; if
test-set metrics are still close to baseline, the model is doing Clever Hans
inference using only the un-noised side and label-distribution structure.

Matrix (configurable in DATASETS / SIDES / SEEDS at the top of this file):
    datasets × sides × seeds  →  one trained model + one test-metrics record per cell.

Only final test metrics are persisted. No epoch-level logs, no checkpoints.
Partial progress is saved to --out after every cell so the run is resumable.

Usage
-----
    python train_clever_hans.py --epochs 80
    python train_clever_hans.py --epochs 30 --datasets DrugBank Drug_CS  # subset
    python train_clever_hans.py --fresh                                    # ignore existing JSON
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from tqdm import tqdm

from config.cfg import get_cfg_defaults
from dataset import MyDataset, NoisyDataset, collate_fn
from model import Model
from processing.clever_hans_utils import get_or_compute_stats


DATASETS: dict[str, dict[str, str]] = {
    "DrugBank": {
        "train":       "lists/db_new/db_train.csv",
        "val":         "lists/db_new/db_val.csv",
        "test":        "lists/db_new/db_test.csv",
        "drug_dir":    "drug/embeddings_atomic",
        "protein_dir": "embeddings",
    },
    "KIBA": {
        "train":       "lists/KIBA/KIBA_pairs_train_stratified.csv",
        "val":         "lists/KIBA/KIBA_pairs_val_stratified.csv",
        "test":        "lists/KIBA/KIBA_pairs_test_stratified.csv",
        "drug_dir":    "drug/embeddings_atomic_KIBA",
        "protein_dir": "embeddings",
    },
    "Drug_CS": {
        "train":       "lists/db_new/db_drug_cs_train.csv",
        "val":         "lists/db_new/db_drug_cs_val.csv",
        "test":        "lists/db_new/db_drug_cs_test.csv",
        "drug_dir":    "drug/embeddings_atomic",
        "protein_dir": "embeddings",
    },
    "Protein_CS": {
        "train":       "lists/db_new/db_protein_cs_train.csv",
        "val":         "lists/db_new/db_protein_cs_val.csv",
        "test":        "lists/db_new/db_protein_cs_test.csv",
        "drug_dir":    "drug/embeddings_atomic",
        "protein_dir": "embeddings",
    },
}

SIDES: list[str] = ["drug", "protein"]
SEEDS: list[int] = [42, 43, 44, 45, 46]


def _metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
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


def _train_one_epoch(model, loader, optim, device) -> float:
    model.train()
    total = 0.0
    n_batches = 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=non_blocking)
        prot_mask = batch["protein_mask"].to(device, non_blocking=non_blocking)
        drug_mask = batch["drug_mask"].to(device, non_blocking=non_blocking)
        prot_emb = batch["protein_emb"].to(device, non_blocking=non_blocking)
        drug_emb = batch["drug_emb"].to(device, non_blocking=non_blocking)

        logits = model(prot_emb, drug_emb,
                       protein_mask=prot_mask, drug_mask=drug_mask, mode="train")
        loss = F.cross_entropy(logits, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    preds_a, labels_a, probs_a = [], [], []
    non_blocking = device.type == "cuda"
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=non_blocking)
        prot_mask = batch["protein_mask"].to(device, non_blocking=non_blocking)
        drug_mask = batch["drug_mask"].to(device, non_blocking=non_blocking)
        prot_emb = batch["protein_emb"].to(device, non_blocking=non_blocking)
        drug_emb = batch["drug_emb"].to(device, non_blocking=non_blocking)
        logits = model(prot_emb, drug_emb,
                       protein_mask=prot_mask, drug_mask=drug_mask, mode="test")
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        labs = labels.detach().cpu().numpy().astype(int)
        preds_a.append(pred); labels_a.append(labs); probs_a.append(probs)
    return _metrics(np.concatenate(labels_a), np.concatenate(preds_a), np.concatenate(probs_a))


def _build_loader(ds, cfg, num_workers: int, device, *, shuffle: bool, drop_last: bool) -> DataLoader:
    kwargs = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle":    shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "drop_last":  drop_last,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def run_cell(
    ds_name: str,
    ds_cfg: dict,
    side: str,
    seed: int,
    *,
    epochs: int,
    num_workers: int,
    cache_dir: Path,
    device: torch.device,
    epoch_pbar_position: int = 1,
) -> dict:
    """Train one (dataset, side, seed) cell and return final test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = get_cfg_defaults()
    cfg.DATA.TRAIN_CSV_PATH = ds_cfg["train"]
    cfg.DATA.VAL_CSV_PATH   = ds_cfg["val"]
    cfg.DATA.TEST_CSV_PATH  = ds_cfg["test"]
    cfg.DATA.DRUG_DIR       = ds_cfg["drug_dir"]
    cfg.DATA.PROTEIN_DIR    = ds_cfg["protein_dir"]

    probe = MyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    trim = probe.trim_terminal_token
    allow_uni = probe.allow_unimol_suffix
    del probe

    stats = get_or_compute_stats(
        cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR,
        tag=ds_name, cache_dir=cache_dir,
        trim_terminal=trim, allow_unimol_suffix=allow_uni,
        verbose=False,
    )

    common = dict(
        noise_side=side,
        drug_mean=stats["drug_mean"], drug_std=stats["drug_std"],
        prot_mean=stats["prot_mean"], prot_std=stats["prot_std"],
        dataset_hint="auto",
        prot_cache_size=4096 if num_workers == 0 else 0,
        drug_cache_size=4096 if num_workers == 0 else 0,
    )
    train_ds = NoisyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR, **common)
    val_ds   = NoisyDataset(cfg.DATA.VAL_CSV_PATH,   cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR, **common)
    test_ds  = NoisyDataset(cfg.DATA.TEST_CSV_PATH,  cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR, **common)

    train_dl = _build_loader(train_ds, cfg, num_workers, device, shuffle=True,  drop_last=True)
    val_dl   = _build_loader(val_ds,   cfg, num_workers, device, shuffle=False, drop_last=False)
    test_dl  = _build_loader(test_ds,  cfg, num_workers, device, shuffle=False, drop_last=False)

    model = Model(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    desc = f"{ds_name}/{side}/s{seed}"
    pbar = tqdm(range(epochs), desc=desc, position=epoch_pbar_position, leave=False, ncols=120)
    for _ in pbar:
        train_loss = _train_one_epoch(model, train_dl, optim, device)
        val_m = _evaluate(model, val_dl, device)
        pbar.set_postfix(
            loss=f"{train_loss:.4f}",
            val_acc=f"{val_m['ACC']:.2f}",
            val_auc=f"{val_m['AUROC']:.4f}",
        )
    pbar.close()

    test_m = _evaluate(model, test_dl, device)

    del model, optim, train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return test_m


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clever Hans full-matrix retraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--sides", type=str, nargs="+", choices=["drug", "protein"], default=SIDES)
    p.add_argument("--datasets", type=str, nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 keeps the dataset's lru_cache (faster on macOS); "
                        ">0 forces caches off so workers can pickle the dataset")
    p.add_argument("--out", type=Path, default=Path("stats/clever_hans/results.json"))
    p.add_argument("--cache_dir", type=Path, default=Path("stats/clever_hans"))
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing --out file and start over")
    args = p.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Sides:    {args.sides}")
    print(f"Seeds:    {args.seeds}")
    print(f"Epochs:   {args.epochs}")
    print(f"Output:   {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.fresh:
        with open(args.out) as f:
            results = json.load(f)
        print(f"Resuming from existing results ({sum(len(v) for v in results.values())} cells already done)")
    else:
        results = {}

    matrix = [
        (ds, side, seed)
        for ds in args.datasets
        for side in args.sides
        for seed in args.seeds
    ]

    pbar = tqdm(matrix, desc="matrix", position=0, ncols=120)
    for ds_name, side, seed in pbar:
        key = f"{ds_name}::{side}"
        existing = results.get(key, [])
        if any(r.get("seed") == seed for r in existing):
            pbar.set_postfix(skip=f"{ds_name}/{side}/s{seed}")
            continue

        pbar.set_postfix(cell=f"{ds_name}/{side}/s{seed}")
        try:
            test_m = run_cell(
                ds_name=ds_name,
                ds_cfg=DATASETS[ds_name],
                side=side,
                seed=seed,
                epochs=args.epochs,
                num_workers=args.num_workers,
                cache_dir=args.cache_dir,
                device=device,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] {ds_name}/{side}/s{seed}: {type(e).__name__}: {e}")
            continue

        record = {"seed": seed, **test_m}
        results.setdefault(key, []).append(record)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

        tqdm.write(
            f"[done] {ds_name}/{side}/s{seed}  "
            f"ACC={test_m['ACC']:.2f}  AUROC={test_m['AUROC']:.4f}  AUPRC={test_m['AUPRC']:.4f}  "
            f"MCC={test_m['MCC']:.4f}"
        )

    pbar.close()
    print(f"\nMatrix complete. Results: {args.out}")


if __name__ == "__main__":
    main()
