"""Find the optimal probability threshold on a validation split, then evaluate on test."""

import argparse
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from config.cfg import get_cfg_defaults
from dataset import MyDataset, collate_fn
from model import Model
from solver import Solver


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def make_loader(csv_path, cfg, dataset_cls, num_workers=4):
    ds = dataset_cls(csv_path, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    kwargs = dict(
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=DEVICE.type == "cuda",
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def collect_probs(solver, loader):
    with torch.no_grad():
        _, _, pred_pairs, _, _, prob_list, _, _ = solver.predict_test(loader)
    pred_pairs = np.asarray(pred_pairs)
    labels = pred_pairs[:, 1].astype(int)
    probs = np.asarray(prob_list, dtype=float)
    return labels, probs


def sweep_threshold(labels, probs, metric="f1", n_steps=1001):
    """Return (best_threshold, best_score, full_curve)."""
    # Use unique probabilities + a fine grid for a complete sweep.
    grid = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, n_steps),
        probs,
    ]))
    grid = np.clip(grid, 0.0, 1.0)

    best_t, best_s = 0.5, -np.inf
    curve = []
    for t in grid:
        preds = (probs >= t).astype(int)
        if metric == "f1":
            s = f1_score(labels, preds, zero_division=0)
        elif metric == "mcc":
            s = matthews_corrcoef(labels, preds) if len(np.unique(preds)) > 1 else 0.0
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            s = tpr - fpr
        elif metric == "accuracy":
            s = accuracy_score(labels, preds)
        else:
            raise ValueError(f"unknown metric {metric}")
        curve.append((t, s))
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s, np.array(curve)


def report(labels, probs, threshold, name):
    preds = (probs >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds) if len(np.unique(preds)) > 1 else 0.0,
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
    }
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    print(f"\n=== {name} @ threshold={threshold:.4f} ===")
    print(f"  Acc:  {metrics['accuracy']:.4f}   MCC: {metrics['mcc']:.4f}")
    print(f"  F1:   {metrics['f1']:.4f}   P:   {metrics['precision']:.4f}   R: {metrics['recall']:.4f}")
    print(f"  AUC:  {metrics['auc']:.4f}   AUPRC: {metrics['auprc']:.4f}")
    print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best_models/model_7607792504713019690_epoch_96.pt")
    p.add_argument("--val", default="lists/db_new/db_val.csv")
    p.add_argument("--test", default="lists/db_new/db_test.csv")
    p.add_argument("--metric", default="f1", choices=["f1", "mcc", "youden", "accuracy", "recall"])
    p.add_argument("--dataset", default="my", choices=["my", "kiba"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-curve", default=None, help="optional path to save sweep curve as .npy")
    args = p.parse_args()

    cfg = get_cfg_defaults()
    dataset_cls = KIBADataset if args.dataset == "kiba" else MyDataset

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = Model(cfg=cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    solver = Solver(model, cfg, device=DEVICE, optim=torch.optim.Adam,
                    loss_fn=cfg.SOLVER.LOSS_FN, eval=None)
    solver.model.eval()

    print(f"\nScoring validation set: {args.val}")
    val_loader = make_loader(args.val, cfg, dataset_cls, args.num_workers)
    val_labels, val_probs = collect_probs(solver, val_loader)
    print(f"  N={len(val_labels)}  pos rate={val_labels.mean():.4f}")

    print(f"\nSweeping thresholds (optimize {args.metric})...")
    best_t, best_s, curve = sweep_threshold(val_labels, val_probs, metric=args.metric)
    print(f"  Best val {args.metric} = {best_s:.4f} at threshold = {best_t:.4f}")

    if args.save_curve:
        np.save(args.save_curve, curve)
        print(f"  Curve saved to {args.save_curve}")

    # Baseline @ 0.5 vs tuned threshold on validation
    report(val_labels, val_probs, 0.5, "VAL @ 0.5")
    report(val_labels, val_probs, best_t, f"VAL @ tuned ({args.metric})")

    print(f"\nScoring test set: {args.test}")
    test_loader = make_loader(args.test, cfg, dataset_cls, args.num_workers)
    test_labels, test_probs = collect_probs(solver, test_loader)
    print(f"  N={len(test_labels)}  pos rate={test_labels.mean():.4f}")

    report(test_labels, test_probs, 0.5, "TEST @ 0.5")
    report(test_labels, test_probs, best_t, f"TEST @ tuned ({args.metric})")


if __name__ == "__main__":
    main()
