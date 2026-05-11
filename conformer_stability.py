"""
Conformer stability analysis for 3DICE.

For a stratified sample of drugs from the DrugBank test set (binned by
rotatable-bond count: ≤3, 4–8, ≥9), generates K conformers per drug via
ETKDG + MMFF94 with varying random seeds, obtains Uni-Mol embeddings for
each, runs inference through the frozen 3DICE model, and reports stability
metrics (prediction flip rate, probability std/range, etc.) per bin.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from tqdm import tqdm

from config.cfg import get_cfg_defaults
from model import Model

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "mps"
)

# User-provided original overall metrics (training baseline run)
ORIGINAL_OVERALL_METRICS = {
    "acc": 0.81984,
    "mcc": 0.64030,
    "auroc": 0.89329,
}


def rotatable_bond_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def bin_label(n_rot: int) -> str:
    if n_rot <= 3:
        return "<=3"
    elif n_rot <= 8:
        return "4-8"
    else:
        return ">=9"


def generate_conformer(smiles: str, seed: int):
    """Generate a single 3D conformer with ETKDG + MMFF94.

    Returns (atoms, coords) where atoms is list[str] and coords is np.ndarray
    of shape (N, 3), or (None, None) on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id < 0:
        return None, None

    # MMFF94 optimisation (fall back to UFF).
    # Use force-field construction directly for compatibility across RDKit versions.
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if mmff_props is not None:
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
            if ff is not None:
                ff.Minimize(maxIts=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
    else:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    conf = mol.GetConformer(conf_id)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                      dtype=np.float32)
    return atoms, coords


def embed_conformer(clf, atoms, coords):
    """Get Uni-Mol atomic representations for a single conformer.

    Returns tensor of shape (L, 512).
    """
    data = {
        "atoms": [atoms],
        "coordinates": [coords],
    }
    f_out, f_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        out = clf.get_repr(data, return_atomic_reprs=True)
    arr = np.asarray(out["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
    emb = torch.from_numpy(arr)
    # Remove SEP token (last row) if present — matches dataset.py convention
    if emb.shape[0] > 1:
        emb = emb[:-1]
    return emb


def load_protein_emb(protein_dir: str, uniprot_id: str):
    """Load the protein embedding for a given uniprot_id (mirrors dataset.py)."""
    import glob as _glob
    pattern = os.path.join(protein_dir, uniprot_id, "*.pt")
    files = sorted(_glob.glob(pattern))
    if not files:
        return None
    emb = torch.load(files[-1], map_location="cpu", weights_only=True)
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    emb = emb.to(dtype=torch.float32).contiguous()
    # Remove SEP
    if emb.shape[0] > 1:
        emb = emb[:-1]
    return emb


def run_inference(model, protein_emb, drug_emb, device):
    """Run a single forward pass. Returns predicted class and P(y=1)."""
    # Add batch dimension
    p = protein_emb.unsqueeze(0).to(device)
    d = drug_emb.unsqueeze(0).to(device)
    p_mask = torch.zeros(1, p.shape[1], dtype=torch.bool, device=device)
    d_mask = torch.zeros(1, d.shape[1], dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = model(p, d, protein_mask=p_mask, drug_mask=d_mask, mode="test")
        probs = F.softmax(logits, dim=1)
    prob_pos = probs[0, 1].item()
    pred = int(probs.argmax(dim=1).item())
    return pred, prob_pos


def compute_binary_mcc(y_true, y_pred):
    """Compute binary Matthews correlation coefficient.

    Returns None if MCC is undefined (denominator is zero).
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return float((tp * tn - fp * fn) / denom)


def compute_binary_auroc(y_true, y_score):
    """Compute binary AUROC from labels and positive-class scores.

    Returns None if AUROC is undefined (only one class present).
    """
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    sum_pos_ranks = sum(ranks[idx] for idx, y in enumerate(y_true) if y == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def bootstrap_metric_values(y_true, y_pred, y_score, metric: str, n_boot: int = 200, seed: int = 42):
    """Return bootstrap samples for a metric over pair records.

    metric must be one of: "acc", "mcc", "auroc".
    """
    n = len(y_true)
    if n < 2:
        return []

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        yt = [y_true[i] for i in sample_idx]
        yp = [y_pred[i] for i in sample_idx]
        ys = [y_score[i] for i in sample_idx]

        if metric == "acc":
            val = float(np.mean([int(t == p) for t, p in zip(yt, yp)]))
        elif metric == "mcc":
            val = compute_binary_mcc(yt, yp)
        elif metric == "auroc":
            val = compute_binary_auroc(yt, ys)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if val is not None:
            values.append(val)

    return values


def bootstrap_metric_std(y_true, y_pred, y_score, metric: str, n_boot: int = 200, seed: int = 42):
    """Estimate metric standard deviation via bootstrap over pairs.

    metric must be one of: "acc", "mcc", "auroc".
    Returns None when insufficient valid bootstrap samples are available.
    """
    values = bootstrap_metric_values(y_true, y_pred, y_score, metric=metric,
                                     n_boot=n_boot, seed=seed)

    if len(values) < 2:
        return None
    return float(np.std(values))


def one_sample_t_test(values, mu0: float = 0.0):
    """Compute two-sided one-sample t-test for mean(values) == mu0.

    Returns dict with t, p, df, n, mean; values may be None if undefined.
    """
    if values is None or len(values) < 2:
        return {"t": None, "p": None, "df": None, "n": len(values or []), "mean": None}

    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1))
    df = n - 1

    if std_v == 0.0:
        if mean_v == mu0:
            return {"t": 0.0, "p": 1.0, "df": df, "n": n, "mean": mean_v}
        sign = 1.0 if mean_v > mu0 else -1.0
        return {"t": sign * float("inf"), "p": 0.0, "df": df, "n": n, "mean": mean_v}

    from scipy.stats import ttest_1samp

    test = ttest_1samp(arr, popmean=mu0, alternative="two-sided")
    t_val = float(test.statistic)
    p_val = float(test.pvalue)
    return {"t": t_val, "p": p_val, "df": df, "n": n, "mean": mean_v}


def compute_perturbation_stats(records, conf_seeds):
    """Compute per-pair perturbation statistics vs the baseline conformer.

    For each (drug, protein) pair that has the baseline seed, computes:
    - mean signed delta P(y=1) across perturbed conformers vs baseline
    - mean absolute delta (perturbation impact magnitude)
    - std of P(y=1) across all conformers (perturbation variability)
    - per-pair flip rate (fraction of perturbed conformers that changed prediction)

    Returns dict of per-pair lists suitable for statistical tests.
    """
    empty = {
        "per_pair_mean_deltas": [],
        "per_pair_mean_abs_deltas": [],
        "per_pair_prob_stds": [],
        "per_pair_flip_rates": [],
    }
    if not records or len(conf_seeds) < 2:
        return empty

    baseline_seed = int(conf_seeds[0])
    mean_deltas = []
    mean_abs_deltas = []
    prob_stds = []
    flip_rates = []

    for r in records:
        seeds_used = r.get("seeds_used", [])
        if baseline_seed not in seeds_used:
            continue

        i_base = seeds_used.index(baseline_seed)
        prob_base = float(r["probs"][i_base])
        pred_base = int(r["preds"][i_base])

        deltas = []
        flips = 0
        for seed in conf_seeds[1:]:
            seed_i = int(seed)
            if seed_i not in seeds_used:
                continue
            i_seed = seeds_used.index(seed_i)
            deltas.append(float(r["probs"][i_seed]) - prob_base)
            if int(r["preds"][i_seed]) != pred_base:
                flips += 1

        if deltas:
            mean_deltas.append(float(np.mean(deltas)))
            mean_abs_deltas.append(float(np.mean([abs(d) for d in deltas])))
            prob_stds.append(float(np.std(r["probs"])))
            flip_rates.append(flips / len(deltas))

    return {
        "per_pair_mean_deltas": mean_deltas,
        "per_pair_mean_abs_deltas": mean_abs_deltas,
        "per_pair_prob_stds": prob_stds,
        "per_pair_flip_rates": flip_rates,
    }


def compute_seed_level_performance(records, conf_seeds, baseline_metrics=None):
    """Compute per-seed metrics and deltas vs baseline metrics.

    If baseline_metrics is provided (keys: acc, mcc, auroc), deltas are
    metric(seed) - baseline_metric and one-sample tests can be run on those
    delta vectors without using any seed as the baseline.
    """
    baseline_metrics = baseline_metrics or {}
    if not records or not conf_seeds:
        return {
            "baseline_metrics": baseline_metrics,
            "per_seed": [],
            "delta_acc": [],
            "delta_mcc": [],
            "delta_auroc": [],
        }

    per_seed = []
    delta_acc = []
    delta_mcc = []
    delta_auroc = []

    mu_acc = baseline_metrics.get("acc")
    mu_mcc = baseline_metrics.get("mcc")
    mu_auroc = baseline_metrics.get("auroc")

    for seed in conf_seeds:
        seed_i = int(seed)
        y_true = []
        y_pred = []
        y_score = []
        for r in records:
            seeds_used = r.get("seeds_used", [])
            if seed_i not in seeds_used:
                continue
            idx = seeds_used.index(seed_i)
            y_true.append(int(r["label"]))
            y_pred.append(int(r["preds"][idx]))
            y_score.append(float(r["probs"][idx]))

        if not y_true:
            continue

        acc_seed = float(np.mean([int(t == p) for t, p in zip(y_true, y_pred)]))
        mcc_seed = compute_binary_mcc(y_true, y_pred)
        auroc_seed = compute_binary_auroc(y_true, y_score)

        d_acc = None
        d_mcc = None
        d_auroc = None

        if mu_acc is not None:
            d_acc = float(acc_seed - mu_acc)
            delta_acc.append(d_acc)

        if mu_mcc is not None and mcc_seed is not None:
            d_mcc = float(mcc_seed - mu_mcc)
            delta_mcc.append(d_mcc)

        if mu_auroc is not None and auroc_seed is not None:
            d_auroc = float(auroc_seed - mu_auroc)
            delta_auroc.append(d_auroc)

        per_seed.append({
            "seed": seed_i,
            "n_pairs": len(y_true),
            "acc": acc_seed,
            "mcc": mcc_seed,
            "auroc": auroc_seed,
            "acc_delta_vs_baseline": d_acc,
            "mcc_delta_vs_baseline": d_mcc,
            "auroc_delta_vs_baseline": d_auroc,
        })

    return {
        "baseline_metrics": baseline_metrics,
        "per_seed": per_seed,
        "delta_acc": delta_acc,
        "delta_mcc": delta_mcc,
        "delta_auroc": delta_auroc,
    }


def summarize_delta_test(values):
    """Return one-sample t-test summary for a delta vector (H0: mean=0)."""
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "t": None,
            "p": None,
            "df": None,
            "effect_size_dz": None,
        }

    mean_v = float(np.mean(values))
    std_v = float(np.std(values, ddof=1)) if n >= 2 else None
    test = one_sample_t_test(values, mu0=0.0)

    ci_low = None
    ci_high = None
    if n >= 2 and std_v is not None:
        try:
            from scipy.stats import t as student_t
            q = float(student_t.ppf(0.975, n - 1))
        except Exception:
            q = 1.96
        half = q * std_v / np.sqrt(n)
        ci_low = mean_v - half
        ci_high = mean_v + half

    dz = None
    if std_v is not None and std_v > 0:
        dz = mean_v / std_v

    return {
        "n": n,
        "mean": mean_v,
        "std": std_v,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t": test["t"],
        "p": test["p"],
        "df": test["df"],
        "effect_size_dz": dz,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conformer stability analysis")
    parser.add_argument("--ckpt", type=str,
                        default="saved/model_6044491872843893979_epoch_50.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--test-csv", type=str,
                        default="lists/db_new/db_test.csv",
                        help="Path to DrugBank test CSV")
    parser.add_argument("--protein-dir", type=str, default="embeddings",
                        help="Protein embedding directory")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of conformers per drug")
    parser.add_argument("--samples-per-bin", type=int, default=30,
                        help="Number of drugs to sample per rotatable-bond bin")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for drug sampling")
    parser.add_argument("--out", type=str, default="conformer_stability_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # ---- Load model -------------------------------------------------------
    if not os.path.exists(args.ckpt):
        # Try saved/ directory
        alt = os.path.join("saved", os.path.basename(args.ckpt))
        if os.path.exists(alt):
            args.ckpt = alt
        else:
            print(f"Checkpoint not found: {args.ckpt}")
            sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = get_cfg_defaults()
    model = Model(cfg=cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {args.ckpt} on {DEVICE}")

    # ---- Load Uni-Mol -----------------------------------------------------
    from unimol_tools import UniMolRepr
    clf = UniMolRepr(data_type="molecule", remove_hs=False)

    # ---- Read test set & build drug table ---------------------------------
    df = pd.read_csv(args.test_csv)
    # Deduplicate to unique drugs
    drug_df = df[["drug_id", "SMILES"]].drop_duplicates(subset=["drug_id"]).copy()
    drug_df = drug_df.dropna(subset=["SMILES"])
    drug_df = drug_df[drug_df["SMILES"].str.strip().astype(bool)]
    drug_df["n_rot"] = drug_df["SMILES"].apply(rotatable_bond_count)
    drug_df = drug_df[drug_df["n_rot"] >= 0]
    drug_df["bin"] = drug_df["n_rot"].apply(bin_label)

    print(f"\nUnique drugs with valid SMILES: {len(drug_df)}")
    print("Distribution by rotatable-bond bin:")
    print(drug_df["bin"].value_counts().sort_index().to_string())

    # ---- Stratified sampling ----------------------------------------------
    rng = np.random.default_rng(args.seed)
    sampled = []
    for b in ["<=3", "4-8", ">=9"]:
        pool = drug_df[drug_df["bin"] == b]
        n = min(args.samples_per_bin, len(pool))
        chosen = pool.sample(n=n, random_state=int(rng.integers(1 << 31)))
        sampled.append(chosen)
        print(f"  Bin {b}: sampled {n}/{len(pool)} drugs")
    sampled_df = pd.concat(sampled)

    # Build mapping: drug_id -> list of (uniprot_id, label) pairs in test set
    pair_map = {}
    for _, row in df.iterrows():
        did = row["drug_id"]
        if did in sampled_df["drug_id"].values:
            pair_map.setdefault(did, []).append((row["uniprot_id"], int(row["interaction"])))

    # ---- Conformer generation seeds ---------------------------------------
    conf_seeds = [0x676767 + i * 6767 for i in range(args.K)]

    # Pre-count total work for progress estimation
    total_pairs = sum(len(pair_map.get(did, [])) for did in sampled_df["drug_id"])
    total_drugs = len(sampled_df)
    print(f"\nWork estimate: {total_drugs} drugs, {total_pairs} (drug, protein) pairs, "
          f"{args.K} conformers each")
    print(f"Total forward passes: ~{total_pairs * args.K}")

    # ---- Run experiment ---------------------------------------------------
    results_per_drug = []
    protein_cache = {}
    t_start = time.perf_counter()

    # Phase-level tracking
    drugs_done = 0
    drugs_skipped = 0
    pairs_done = 0
    conformer_failures = 0
    protein_misses = 0
    conf_gen_time = 0.0
    unimol_time = 0.0
    inference_time = 0.0

    drug_pbar = tqdm(sampled_df.iterrows(), total=total_drugs, desc="Drugs",
                     unit="drug", dynamic_ncols=True)
    for _, drug_row in drug_pbar:
        drug_id = drug_row["drug_id"]
        smiles = drug_row["SMILES"]
        n_rot = drug_row["n_rot"]
        b = drug_row["bin"]

        # Generate K conformer embeddings
        conformer_embs = []
        conformer_emb_seeds = []
        t0 = time.perf_counter()
        for seed in conf_seeds:
            atoms, coords = generate_conformer(smiles, seed)
            if atoms is None:
                conformer_failures += 1
                continue
            try:
                t_u0 = time.perf_counter()
                emb = embed_conformer(clf, atoms, coords)
                unimol_time += time.perf_counter() - t_u0
                conformer_embs.append(emb)
                conformer_emb_seeds.append(int(seed))
            except Exception as e:
                conformer_failures += 1
                tqdm.write(f"  Uni-Mol failed for {drug_id} seed={seed}: {e}")
                continue
        conf_gen_time += time.perf_counter() - t0

        if len(conformer_embs) < 2:
            drugs_skipped += 1
            tqdm.write(f"  Skipping {drug_id}: only {len(conformer_embs)} conformer(s) succeeded")
            drugs_done += 1
            drug_pbar.set_postfix(skip=drugs_skipped, pairs=pairs_done, ordered_dict=None)
            continue

        # For each (drug, protein) pair, run inference with each conformer
        pairs = pair_map.get(drug_id, [])
        for uniprot_id, label in pairs:
            # Load protein embedding (cached)
            if uniprot_id not in protein_cache:
                prot_emb = load_protein_emb(args.protein_dir, uniprot_id)
                if prot_emb is None:
                    protein_misses += 1
                    continue
                protein_cache[uniprot_id] = prot_emb
            prot_emb = protein_cache[uniprot_id]

            preds = []
            probs = []
            t_inf0 = time.perf_counter()
            for drug_emb in conformer_embs:
                pred, prob_pos = run_inference(model, prot_emb, drug_emb, DEVICE)
                preds.append(pred)
                probs.append(prob_pos)
            inference_time += time.perf_counter() - t_inf0

            pairs_done += 1
            results_per_drug.append({
                "drug_id": drug_id,
                "uniprot_id": uniprot_id,
                "label": label,
                "bin": b,
                "n_rot": int(n_rot),
                "n_conformers": len(conformer_embs),
                "seeds_used": list(conformer_emb_seeds),
                "preds": preds,
                "probs": probs,
            })

        drugs_done += 1
        elapsed = time.perf_counter() - t_start
        rate = drugs_done / elapsed
        eta = (total_drugs - drugs_done) / rate if rate > 0 else 0
        drug_pbar.set_postfix_str(
            f"pairs={pairs_done} | skip={drugs_skipped} | "
            f"ETA {eta:.0f}s | conf {conf_gen_time:.1f}s | "
            f"unimol {unimol_time:.1f}s | infer {inference_time:.1f}s"
        )
    drug_pbar.close()

    wall_time = time.perf_counter() - t_start
    print(f"\nCompleted in {wall_time:.1f}s")
    print(f"  Drugs processed: {drugs_done} ({drugs_skipped} skipped due to conformer failures)")
    print(f"  Conformer failures: {conformer_failures}/{total_drugs * args.K} attempts")
    print(f"  Protein embedding misses: {protein_misses}")
    print(f"  Time breakdown: conformer+unimol {conf_gen_time:.1f}s "
          f"(unimol alone {unimol_time:.1f}s) | inference {inference_time:.1f}s | "
          f"overhead {wall_time - conf_gen_time - inference_time:.1f}s")

    # ---- Compute metrics --------------------------------------------------
    print(f"\nTotal (drug, protein) pairs evaluated: {len(results_per_drug)}")

    def compute_metrics(records):
        if not records:
            return {}
        flip_rates = []
        prob_stds = []
        prob_ranges = []
        majority_accs = []
        y_true = []
        y_pred_majority = []
        y_score_mean = []
        y_true_baseline = []
        y_pred_baseline = []
        y_score_baseline = []
        baseline_seed = conf_seeds[0]

        for r in records:
            preds = r["preds"]
            probs = r["probs"]
            label = r["label"]

            # Flip rate: fraction of conformer pairs that disagree
            n = len(preds)
            if n < 2:
                continue
            n_disagree = sum(1 for i in range(n) for j in range(i + 1, n) if preds[i] != preds[j])
            n_pairs = n * (n - 1) // 2
            flip_rates.append(n_disagree / n_pairs)

            prob_stds.append(np.std(probs))
            prob_ranges.append(max(probs) - min(probs))

            # Majority vote accuracy
            majority = 1 if sum(preds) > n / 2 else 0
            majority_accs.append(int(majority == label))
            y_true.append(int(label))
            y_pred_majority.append(int(majority))
            y_score_mean.append(float(np.mean(probs)))

            # Baseline conformer metrics on this sampled subset.
            seeds_used = r.get("seeds_used", [])
            if baseline_seed in seeds_used:
                seed_idx = seeds_used.index(baseline_seed)
                y_true_baseline.append(int(label))
                y_pred_baseline.append(int(preds[seed_idx]))
                y_score_baseline.append(float(probs[seed_idx]))

        mcc = compute_binary_mcc(y_true, y_pred_majority) if y_true else None
        auroc = compute_binary_auroc(y_true, y_score_mean) if y_true else None

        acc_boot = bootstrap_metric_values(y_true, y_pred_majority, y_score_mean,
                                           metric="acc", seed=101) if y_true else []
        mcc_boot = bootstrap_metric_values(y_true, y_pred_majority, y_score_mean,
                                           metric="mcc", seed=202) if y_true else []
        auroc_boot = bootstrap_metric_values(y_true, y_pred_majority, y_score_mean,
                                             metric="auroc", seed=303) if y_true else []

        acc_std = float(np.std(acc_boot)) if len(acc_boot) >= 2 else None
        mcc_std = float(np.std(mcc_boot)) if len(mcc_boot) >= 2 else None
        auroc_std = float(np.std(auroc_boot)) if len(auroc_boot) >= 2 else None

        baseline_acc = float(np.mean([int(t == p) for t, p in zip(y_true_baseline, y_pred_baseline)])) if y_true_baseline else None
        baseline_mcc = compute_binary_mcc(y_true_baseline, y_pred_baseline) if y_true_baseline else None
        baseline_auroc = compute_binary_auroc(y_true_baseline, y_score_baseline) if y_true_baseline else None
        return {
            "n_pairs": len(records),
            "n_drugs": len(set(r["drug_id"] for r in records)),
            "mean_flip_rate": float(np.mean(flip_rates)) if flip_rates else None,
            "mean_prob_std": float(np.mean(prob_stds)) if prob_stds else None,
            "mean_prob_range": float(np.mean(prob_ranges)) if prob_ranges else None,
            "max_prob_range": float(np.max(prob_ranges)) if prob_ranges else None,
            "majority_vote_acc": float(np.mean(majority_accs)) if majority_accs else None,
            "majority_vote_acc_std": acc_std,
            "majority_vote_mcc": mcc,
            "majority_vote_mcc_std": mcc_std,
            "mean_prob_auroc": auroc,
            "mean_prob_auroc_std": auroc_std,
            "baseline_n_pairs": len(y_true_baseline),
            "baseline_acc": baseline_acc,
            "baseline_mcc": baseline_mcc,
            "baseline_auroc": baseline_auroc,
            "_boot_acc": acc_boot,
            "_boot_mcc": mcc_boot,
            "_boot_auroc": auroc_boot,
        }

    overall = compute_metrics(results_per_drug)
    per_bin = {}
    for b in ["<=3", "4-8", ">=9"]:
        bin_records = [r for r in results_per_drug if r["bin"] == b]
        per_bin[b] = compute_metrics(bin_records)

    # Seed-level performance tests (answers: does seed change ACC/MCC/AUROC?).
    def _seed_level_tests(records, baseline_metrics):
        stats = compute_seed_level_performance(records, conf_seeds, baseline_metrics=baseline_metrics)
        return {
            "baseline_metrics": stats["baseline_metrics"],
            "n_seeds": len(stats["per_seed"]),
            "acc_delta": summarize_delta_test(stats["delta_acc"]),
            "mcc_delta": summarize_delta_test(stats["delta_mcc"]),
            "auroc_delta": summarize_delta_test(stats["delta_auroc"]),
            "per_seed": stats["per_seed"],
        }

    seed_level_tests = {
        "overall": _seed_level_tests(results_per_drug, baseline_metrics=ORIGINAL_OVERALL_METRICS),
        "per_bin": {
            b: _seed_level_tests([r for r in results_per_drug if r["bin"] == b], baseline_metrics={})
            for b in ["<=3", "4-8", ">=9"]
        },
    }

    # Bonferroni correction for seed-level performance tests:
    # 3 metrics × (overall + 3 bins) = 12 tests.
    _N_TESTS_SEED_LEVEL = 12

    def _apply_bonferroni_seed_level(seed_tests):
        groups = [seed_tests["overall"]] + [
            seed_tests["per_bin"][b] for b in ["<=3", "4-8", ">=9"]
        ]
        for g in groups:
            for key in ("acc_delta", "mcc_delta", "auroc_delta"):
                sub = g.get(key, {})
                p_raw = sub.get("p")
                sub["p_bonferroni"] = (
                    min(p_raw * _N_TESTS_SEED_LEVEL, 1.0) if p_raw is not None else None
                )

    _apply_bonferroni_seed_level(seed_level_tests)

    # Perturbation impact tests (per-pair, high power).
    # For each pair, computes deltas of perturbed conformers vs baseline:
    #   (1) Signed delta t-test: H0: mean(perturbed - baseline) = 0  (directional bias)
    #   (2) Absolute delta t-test: H0: mean|perturbed - baseline| = 0  (impact magnitude)
    #   (3) Perturbation std t-test: H0: mean(std across conformers) = 0  (variability)
    # All tests use n = N_pairs, df = N_pairs - 1.
    def _perturbation_tests(stats, m):
        if not stats or not m:
            return {}

        deltas = stats["per_pair_mean_deltas"]
        abs_deltas = stats["per_pair_mean_abs_deltas"]
        prob_stds = stats["per_pair_prob_stds"]
        flip_rates = stats["per_pair_flip_rates"]

        delta_test = one_sample_t_test(deltas, mu0=0.0)
        abs_test = one_sample_t_test(abs_deltas, mu0=0.0)
        std_test = one_sample_t_test(prob_stds, mu0=0.0)

        return {
            "baseline_pairs": m.get("baseline_n_pairs"),
            "signed_delta": {
                "description": "H0: mean(perturbed - baseline) = 0 (no directional bias)",
                "n": delta_test["n"],
                "mean": delta_test["mean"],
                "std": float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else None,
                "t": delta_test["t"],
                "p": delta_test["p"],
                "df": delta_test["df"],
            },
            "abs_delta": {
                "description": "H0: mean|perturbed - baseline| = 0 (no perturbation impact)",
                "n": abs_test["n"],
                "mean": abs_test["mean"],
                "std": float(np.std(abs_deltas, ddof=1)) if len(abs_deltas) >= 2 else None,
                "t": abs_test["t"],
                "p": abs_test["p"],
                "df": abs_test["df"],
            },
            "perturbation_std": {
                "description": "H0: mean(std of P(y=1) across conformers) = 0 (no variability)",
                "n": std_test["n"],
                "mean": std_test["mean"],
                "std": float(np.std(prob_stds, ddof=1)) if len(prob_stds) >= 2 else None,
                "t": std_test["t"],
                "p": std_test["p"],
                "df": std_test["df"],
            },
            "flip_rate": {
                "description": "Fraction of perturbed conformers that changed the binary prediction",
                "n": len(flip_rates),
                "mean": float(np.mean(flip_rates)) if flip_rates else None,
                "std": float(np.std(flip_rates, ddof=1)) if len(flip_rates) >= 2 else None,
            },
        }

    overall_pert = compute_perturbation_stats(results_per_drug, conf_seeds)
    per_bin_pert = {
        b: compute_perturbation_stats(
            [r for r in results_per_drug if r["bin"] == b], conf_seeds
        )
        for b in ["<=3", "4-8", ">=9"]
    }

    t_tests = {
        "overall": _perturbation_tests(overall_pert, overall),
        "per_bin": {
            b: _perturbation_tests(per_bin_pert[b], per_bin[b])
            for b in ["<=3", "4-8", ">=9"]
        },
    }

    # Bonferroni correction: 3 test types × 4 groups (overall + 3 bins) = 12 tests.
    _N_TESTS_BONFERRONI = 12

    def _apply_bonferroni(t_tests_dict):
        """Add Bonferroni-corrected p-values in-place."""
        all_groups = [t_tests_dict["overall"]] + [
            t_tests_dict["per_bin"][b] for b in ["<=3", "4-8", ">=9"]
        ]
        for g in all_groups:
            if not g:
                continue
            for test_key in ("signed_delta", "abs_delta", "perturbation_std"):
                sub = g.get(test_key, {})
                p_raw = sub.get("p")
                sub["p_bonferroni"] = (
                    min(p_raw * _N_TESTS_BONFERRONI, 1.0) if p_raw is not None else None
                )

    _apply_bonferroni(t_tests)

    def _strip_bootstrap_fields(metric_dict):
        metric_dict.pop("_boot_acc", None)
        metric_dict.pop("_boot_mcc", None)
        metric_dict.pop("_boot_auroc", None)

    _strip_bootstrap_fields(overall)
    for b in ["<=3", "4-8", ">=9"]:
        _strip_bootstrap_fields(per_bin[b])

    # ---- Report -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFORMER STABILITY ANALYSIS")
    print("=" * 60)

    print(f"\nOverall ({overall.get('n_pairs', 0)} pairs, {overall.get('n_drugs', 0)} drugs):")
    print(f"  Prediction flip rate:  {overall.get('mean_flip_rate', 'N/A'):.5f}"
          if overall.get('mean_flip_rate') is not None else "  Prediction flip rate:  N/A")
    print(f"  Mean P(y=1) std:       {overall.get('mean_prob_std', 'N/A'):.5f}"
          if overall.get('mean_prob_std') is not None else "  Mean P(y=1) std:       N/A")
    print(f"  Mean P(y=1) range:     {overall.get('mean_prob_range', 'N/A'):.5f}"
          if overall.get('mean_prob_range') is not None else "  Mean P(y=1) range:     N/A")
    print(f"  Max P(y=1) range:      {overall.get('max_prob_range', 'N/A'):.5f}"
          if overall.get('max_prob_range') is not None else "  Max P(y=1) range:      N/A")
    print(f"  Majority-vote accuracy:{overall.get('majority_vote_acc', 'N/A'):.5f}"
          if overall.get('majority_vote_acc') is not None else "  Majority-vote accuracy: N/A")
    print(f"  Accuracy std:          {overall.get('majority_vote_acc_std', 'N/A'):.5f}"
          if overall.get('majority_vote_acc_std') is not None else "  Accuracy std:          N/A")
    print(f"  Majority-vote MCC:     {overall.get('majority_vote_mcc', 'N/A'):.5f}"
          if overall.get('majority_vote_mcc') is not None else "  Majority-vote MCC:     N/A")
    print(f"  MCC std:               {overall.get('majority_vote_mcc_std', 'N/A'):.5f}"
          if overall.get('majority_vote_mcc_std') is not None else "  MCC std:               N/A")
    print(f"  AUROC (mean P(y=1)):   {overall.get('mean_prob_auroc', 'N/A'):.5f}"
          if overall.get('mean_prob_auroc') is not None else "  AUROC (mean P(y=1)):   N/A")
    print(f"  AUROC std:             {overall.get('mean_prob_auroc_std', 'N/A'):.5f}"
          if overall.get('mean_prob_auroc_std') is not None else "  AUROC std:             N/A")

    for b in ["<=3", "4-8", ">=9"]:
        m = per_bin[b]
        print(f"\nBin {b} ({m.get('n_pairs', 0)} pairs, {m.get('n_drugs', 0)} drugs):")
        if m.get("mean_flip_rate") is not None:
            print(f"  Prediction flip rate:  {m['mean_flip_rate']:.5f}")
            print(f"  Mean P(y=1) std:       {m['mean_prob_std']:.5f}")
            print(f"  Mean P(y=1) range:     {m['mean_prob_range']:.5f}")
            print(f"  Max P(y=1) range:      {m['max_prob_range']:.5f}")
            print(f"  Majority-vote accuracy:{m['majority_vote_acc']:.5f}")
            print(f"  Accuracy std:          {m['majority_vote_acc_std']:.5f}"
                if m.get('majority_vote_acc_std') is not None else "  Accuracy std:          N/A")
            print(f"  Majority-vote MCC:     {m['majority_vote_mcc']:.5f}"
                if m.get('majority_vote_mcc') is not None else "  Majority-vote MCC:     N/A")
            print(f"  MCC std:               {m['majority_vote_mcc_std']:.5f}"
                if m.get('majority_vote_mcc_std') is not None else "  MCC std:               N/A")
            print(f"  AUROC (mean P(y=1)):   {m['mean_prob_auroc']:.5f}"
                if m.get('mean_prob_auroc') is not None else "  AUROC (mean P(y=1)):   N/A")
            print(f"  AUROC std:             {m['mean_prob_auroc_std']:.5f}"
                if m.get('mean_prob_auroc_std') is not None else "  AUROC std:             N/A")
        else:
            print("  No data")

    print("\nRepresentativeness check (baseline subsample vs published full-set metrics):")
    baseline_acc = overall.get("baseline_acc")
    baseline_mcc = overall.get("baseline_mcc")
    baseline_auroc = overall.get("baseline_auroc")
    for metric_name, sub_val, orig_val in [
        ("acc",   baseline_acc,   ORIGINAL_OVERALL_METRICS["acc"]),
        ("mcc",   baseline_mcc,   ORIGINAL_OVERALL_METRICS["mcc"]),
        ("auroc", baseline_auroc, ORIGINAL_OVERALL_METRICS["auroc"]),
    ]:
        sub_s = f"{sub_val:.5f}" if sub_val is not None else "N/A"
        print(f"  {metric_name}: subsample_baseline={sub_s}, full_testset={orig_val:.5f}")

    print("\nSeed-level performance difference tests (n = seeds):")
    print(f"  Bonferroni correction: n_tests = {_N_TESTS_SEED_LEVEL}")
    print("  H0 for each metric: mean(delta_vs_original_baseline) = 0")
    print("  Overall uses ORIGINAL_OVERALL_METRICS as baseline; per-bin baseline tests are omitted unless bin baselines are provided.")

    def _print_seed_level_block(name, vals):
        if not vals:
            print(f"  {name}: No data")
            return

        def _fmt(v, spec=".5f"):
            return format(v, spec) if v is not None else "N/A"

        baseline_vals = vals.get("baseline_metrics", {})
        print(
            f"  {name} (n_seeds={vals.get('n_seeds', 0)}):"
        )
        if baseline_vals:
            print(
                f"    baseline metrics: acc={_fmt(baseline_vals.get('acc'))}, "
                f"mcc={_fmt(baseline_vals.get('mcc'))}, auroc={_fmt(baseline_vals.get('auroc'))}"
            )

        for key, label in [
            ("acc_delta", "ACC delta"),
            ("mcc_delta", "MCC delta"),
            ("auroc_delta", "AUROC delta"),
        ]:
            sub = vals.get(key, {})
            print(
                f"    {label}: n={sub.get('n', 'N/A')}, "
                f"mean={_fmt(sub.get('mean'))}, std={_fmt(sub.get('std'))}, "
                f"ci95=[{_fmt(sub.get('ci95_low'))}, {_fmt(sub.get('ci95_high'))}], "
                f"t={_fmt(sub.get('t'))}, p={_fmt(sub.get('p'), '.4g')}, "
                f"p_bonf={_fmt(sub.get('p_bonferroni'), '.4g')}, "
                f"dz={_fmt(sub.get('effect_size_dz'))}"
            )

    _print_seed_level_block("overall", seed_level_tests.get("overall", {}))
    for b in ["<=3", "4-8", ">=9"]:
        _print_seed_level_block(f"bin {b}", seed_level_tests.get("per_bin", {}).get(b, {}))

    print("\nPerturbation impact tests (per-pair, n = N_pairs):")
    print(f"  Bonferroni correction: n_tests = {_N_TESTS_BONFERRONI}")

    def _print_t_block(name, vals):
        if not vals:
            print(f"  {name}: No data")
            return

        def _fmt(v, spec=".5f"):
            return format(v, spec) if v is not None else "N/A"

        print(
            f"  {name} (n_baseline_pairs={vals.get('baseline_pairs')}):"
        )
        for test_key, label in [
            ("signed_delta", "Signed delta (bias)"),
            ("abs_delta", "Abs delta (impact)"),
            ("perturbation_std", "Perturbation std"),
        ]:
            sub = vals.get(test_key, {})
            print(
                f"    {label}: n={sub.get('n', 'N/A')}, "
                f"mean={_fmt(sub.get('mean'))}, std={_fmt(sub.get('std'))}, "
                f"t={_fmt(sub.get('t'))}, p={_fmt(sub.get('p'), '.4g')}, "
                f"p_bonf={_fmt(sub.get('p_bonferroni'), '.4g')}"
            )
        flip = vals.get("flip_rate", {})
        print(
            f"    Flip rate: n={flip.get('n', 'N/A')}, "
            f"mean={_fmt(flip.get('mean'))}, std={_fmt(flip.get('std'))}"
        )

    _print_t_block("overall", t_tests.get("overall", {}))
    for b in ["<=3", "4-8", ">=9"]:
        _print_t_block(f"bin {b}", t_tests.get("per_bin", {}).get(b, {}))

    # ---- Save full results ------------------------------------------------
    output = {
        "config": {
            "checkpoint": args.ckpt,
            "test_csv": args.test_csv,
            "K": args.K,
            "samples_per_bin": args.samples_per_bin,
            "conformer_seeds": conf_seeds,
            "seed": args.seed,
        },
        "representativeness_check": {
            "note": (
                "Compares baseline subsample metrics against published full test-set metrics. "
                "Differences reflect subsampling variance, NOT conformer instability. "
                "Do NOT use ORIGINAL_OVERALL_METRICS as mu0 in a conformer-stability t-test."
            ),
            "original_overall_metrics": ORIGINAL_OVERALL_METRICS,
            "baseline_subsample_acc": overall.get("baseline_acc"),
            "baseline_subsample_mcc": overall.get("baseline_mcc"),
            "baseline_subsample_auroc": overall.get("baseline_auroc"),
        },
        "timing": {
            "wall_time_s": round(wall_time, 2),
            "conformer_gen_s": round(conf_gen_time, 2),
            "unimol_embed_s": round(unimol_time, 2),
            "inference_s": round(inference_time, 2),
            "conformer_failures": conformer_failures,
            "drugs_skipped": drugs_skipped,
            "protein_misses": protein_misses,
        },
        "overall": overall,
        "per_bin": per_bin,
        "seed_level_tests": seed_level_tests,
        "t_tests": t_tests,
        "per_pair": results_per_drug,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
