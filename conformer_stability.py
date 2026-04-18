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
from rdkit.Chem import AllChem, rdMolDescriptors, RDLogger
from tqdm import tqdm

from config.cfg import get_cfg_defaults
from model import Model

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


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

    # MMFF94 optimisation (fall back to UFF)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is not None:
        AllChem.MMFFOptimizeMolecule(mol, mmff_props)
    else:
        AllChem.UFFOptimizeMolecule(mol)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conformer stability analysis")
    parser.add_argument("--ckpt", type=str,
                        default="best_models/model_-7649301121676988024_epoch_56.pt",
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
    conf_seeds = [0xC0FFEE + i * 1337 for i in range(args.K)]

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

        return {
            "n_pairs": len(records),
            "n_drugs": len(set(r["drug_id"] for r in records)),
            "mean_flip_rate": float(np.mean(flip_rates)) if flip_rates else None,
            "mean_prob_std": float(np.mean(prob_stds)) if prob_stds else None,
            "mean_prob_range": float(np.mean(prob_ranges)) if prob_ranges else None,
            "max_prob_range": float(np.max(prob_ranges)) if prob_ranges else None,
            "majority_vote_acc": float(np.mean(majority_accs)) if majority_accs else None,
        }

    overall = compute_metrics(results_per_drug)
    per_bin = {}
    for b in ["<=3", "4-8", ">=9"]:
        bin_records = [r for r in results_per_drug if r["bin"] == b]
        per_bin[b] = compute_metrics(bin_records)

    # ---- Report -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFORMER STABILITY ANALYSIS")
    print("=" * 60)

    print(f"\nOverall ({overall.get('n_pairs', 0)} pairs, {overall.get('n_drugs', 0)} drugs):")
    print(f"  Prediction flip rate:  {overall.get('mean_flip_rate', 'N/A'):.4f}"
          if overall.get('mean_flip_rate') is not None else "  Prediction flip rate:  N/A")
    print(f"  Mean P(y=1) std:       {overall.get('mean_prob_std', 'N/A'):.4f}"
          if overall.get('mean_prob_std') is not None else "  Mean P(y=1) std:       N/A")
    print(f"  Mean P(y=1) range:     {overall.get('mean_prob_range', 'N/A'):.4f}"
          if overall.get('mean_prob_range') is not None else "  Mean P(y=1) range:     N/A")
    print(f"  Max P(y=1) range:      {overall.get('max_prob_range', 'N/A'):.4f}"
          if overall.get('max_prob_range') is not None else "  Max P(y=1) range:      N/A")
    print(f"  Majority-vote accuracy:{overall.get('majority_vote_acc', 'N/A'):.4f}"
          if overall.get('majority_vote_acc') is not None else "  Majority-vote accuracy: N/A")

    for b in ["<=3", "4-8", ">=9"]:
        m = per_bin[b]
        print(f"\nBin {b} ({m.get('n_pairs', 0)} pairs, {m.get('n_drugs', 0)} drugs):")
        if m.get("mean_flip_rate") is not None:
            print(f"  Prediction flip rate:  {m['mean_flip_rate']:.4f}")
            print(f"  Mean P(y=1) std:       {m['mean_prob_std']:.4f}")
            print(f"  Mean P(y=1) range:     {m['mean_prob_range']:.4f}")
            print(f"  Max P(y=1) range:      {m['max_prob_range']:.4f}")
            print(f"  Majority-vote accuracy:{m['majority_vote_acc']:.4f}")
        else:
            print("  No data")

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
        "per_pair": results_per_drug,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
