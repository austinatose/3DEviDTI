"""Plot Clever Hans results in the performance-heatmap style.

For each dataset (DrugBank, Drug_CS, Protein_CS):
  rows = metrics (ACC, PPV, TPR, F1, MCC, AUROC, AUPRC)
  cols = [3DICE baseline (real both sides), drug-noise retrain, protein-noise retrain]
  cells = mean ± std across seeds; colour is row-normalised
  baseline column is outlined to mark it as the reference.

Inputs
------
  stats/clever_hans/clever_hans_CH.json   (output of train_clever_hans.py)
  stats/results.json                      (existing 3DICE baseline runs)

Outputs
-------
  stat_test_plots/clever_hans/<dataset>_clever_hans.png

Run
---
  python stats/clever_hans_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["ACC", "PPV", "TPR", "F1", "MCC", "AUROC", "AUPRC"]

# Map CH-result key prefix → key in stats/results.json
DATASET_MAP = {
    "DrugBank":   "DrugBank",
    "Drug_CS":    "Drug Cold-start",
    "Protein_CS": "Protein Cold-start",
}

COL_LABELS = ["3DICE\n(real both)", "drug-noise\nretrain", "protein-noise\nretrain"]


def _ch_records_to_array(records: list[dict]) -> np.ndarray:
    """list of dicts → (n_seeds, n_metrics) with ACC normalised to fraction."""
    arr = np.array([[r[m] for m in METRICS] for r in records], dtype=float)
    # train_clever_hans.py stores ACC as percentage; baseline JSON stores as fraction.
    arr[:, 0] /= 100.0
    return arr


def extract_matrix(ch: dict, baseline: dict, ds_key: str, baseline_key: str):
    """Return (mean_mat, std_mat) of shape (n_metrics, 3)."""
    n_met = len(METRICS)
    mean_mat = np.zeros((n_met, 3))
    std_mat = np.zeros((n_met, 3))

    base_runs = np.array(baseline["experiments"][baseline_key]["models"]["3DICE"], dtype=float)
    mean_mat[:, 0] = base_runs.mean(axis=0)
    std_mat[:, 0] = base_runs.std(axis=0, ddof=1)

    drug_arr = _ch_records_to_array(ch[f"{ds_key}::drug"])
    mean_mat[:, 1] = drug_arr.mean(axis=0)
    std_mat[:, 1] = drug_arr.std(axis=0, ddof=1) if drug_arr.shape[0] > 1 else 0.0

    prot_arr = _ch_records_to_array(ch[f"{ds_key}::protein"])
    mean_mat[:, 2] = prot_arr.mean(axis=0)
    std_mat[:, 2] = prot_arr.std(axis=0, ddof=1) if prot_arr.shape[0] > 1 else 0.0

    return mean_mat, std_mat


def plot_one(title: str, mean_mat: np.ndarray, std_mat: np.ndarray, out_path: Path) -> None:
    n_met, n_cols = mean_mat.shape

    row_min = mean_mat.min(axis=1, keepdims=True)
    row_max = mean_mat.max(axis=1, keepdims=True)
    row_range = np.where(row_max - row_min > 0, row_max - row_min, 1.0)
    norm_mat = (mean_mat - row_min) / row_range

    fig_w = max(3.5, 1.7 * n_cols + 1.2)
    fig_h = max(3.0, 0.78 * n_met + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm_mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COL_LABELS, fontsize=9)
    ax.set_yticks(range(n_met))
    ax.set_yticklabels(METRICS, fontsize=9)
    ax.set_title(f"{title}: Clever Hans (mean ± std)", pad=10)

    for r in range(n_met):
        for c in range(n_cols):
            brightness = norm_mat[r, c]
            color = "white" if brightness > 0.6 else "#1a1a1a"
            ax.text(
                c, r,
                f"{mean_mat[r, c]:.4f}\n±{std_mat[r, c]:.4f}",
                ha="center", va="center", fontsize=7, color=color,
            )

    # Bold outline on baseline column (col 0)
    for r in range(n_met):
        ax.add_patch(plt.Rectangle(
            (0 - 0.5, r - 0.5), 1, 1,
            fill=False, edgecolor="#e05c00", linewidth=1.8, zorder=5,
        ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("score", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["min", "mid", "max"], fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ch_path = repo / "stats" / "clever_hans" / "clever_hans_CH.json"
    baseline_path = repo / "stats" / "results.json"
    out_dir = repo / "stat_test_plots" / "clever_hans"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ch_path) as f:
        ch = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"CH results: {ch_path.name}  keys: {list(ch.keys())}")
    print(f"Output dir: {out_dir}")
    print()

    for ds_key, baseline_key in DATASET_MAP.items():
        print(f"=== {baseline_key} ===")
        missing = [k for k in (f"{ds_key}::drug", f"{ds_key}::protein") if k not in ch]
        if missing:
            print(f"  skip: missing CH keys {missing}")
            continue
        if baseline_key not in baseline.get("experiments", {}):
            print(f"  skip: baseline missing key {baseline_key!r}")
            continue

        mean_mat, std_mat = extract_matrix(ch, baseline, ds_key, baseline_key)
        out_path = out_dir / f"{baseline_key.replace(' ', '_')}_clever_hans.png"
        plot_one(baseline_key, mean_mat, std_mat, out_path)
        print(f"  saved {out_path.name}")

        for i, m in enumerate(METRICS):
            print(
                f"    {m:6s}  baseline={mean_mat[i, 0]:.4f}±{std_mat[i, 0]:.4f}   "
                f"drug-noise={mean_mat[i, 1]:.4f}±{std_mat[i, 1]:.4f}   "
                f"protein-noise={mean_mat[i, 2]:.4f}±{std_mat[i, 2]:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
