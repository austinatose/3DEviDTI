import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from dataset import MyDataset, collate_fn, KIBADataset
from model import Model
from config.cfg import get_cfg_defaults
from solver import Solver
import matplotlib.pyplot as plt
import numpy as np

import py3Dmol

import argparse

# ---- config ----
parser = argparse.ArgumentParser(description="Inspect attention for a specific (drug_id, uniprot_id) pair")
parser.add_argument("--ckpt", type=str, default="saved/model_-6508505680996270503_epoch_79.pt", help="Path to checkpoint")
parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda/mps")
parser.add_argument("--csv", type=str, default="lists/db_test.csv", help="CSV file to search")
parser.add_argument("--drug_id", type=str, default=None, help="DrugBank ID to find (e.g., DB00001)")
parser.add_argument("--uniprot_id", type=str, default=None, help="UniProt ID to find (e.g., P00533)")
parser.add_argument("--max_search", type=int, default=20000, help="Max batches to scan for the pair")
args = parser.parse_args()

CKPT_PATH = args.ckpt
DEVICE = args.device

# ---- load checkpoint ----
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
cfg = get_cfg_defaults()

model = Model(cfg=cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)

solver = Solver(model, cfg, device=DEVICE, optim=torch.optim.Adam, loss_fn=cfg.SOLVER.LOSS_FN, eval=None)

test_ds = MyDataset(args.csv, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)

# ---- find a specific (drug_id, uniprot_id) pair if requested ----
selected = None
if args.drug_id is not None or args.uniprot_id is not None:
    print(f"Searching for pair: drug_id={args.drug_id}, uniprot_id={args.uniprot_id} in {args.csv}")
    for i, batch in enumerate(test_dl):
        if i >= args.max_search:
            break
        b_drug = batch.get("drug_id", [None])[0]
        b_prot = batch.get("uniprot_id", [None])[0]

        ok_drug = (args.drug_id is None) or (b_drug == args.drug_id)
        ok_prot = (args.uniprot_id is None) or (b_prot == args.uniprot_id)

        if ok_drug and ok_prot:
            selected = batch
            print(f"Found at batch {i}: drug_id={b_drug}, uniprot_id={b_prot}")
            break

    if selected is None:
        raise RuntimeError(
            "Pair not found. Check IDs, or increase --max_search, or confirm the CSV contains that pair. "
            f"Requested drug_id={args.drug_id}, uniprot_id={args.uniprot_id}."
        )
else:
    # fallback: original behavior
    selected = next(iter(test_dl))

set = selected
labels = set["label"].to(DEVICE)
protein_emb = set["protein_emb"].to(DEVICE)
drug_emb = set["drug_emb"].to(DEVICE)
protein_mask = set["protein_mask"].to(DEVICE)
drug_mask = set["drug_mask"].to(DEVICE)
drug_id = set["drug_id"]
protein_id = set["uniprot_id"]
print("Drug ID:", drug_id)
print("Protein ID:", protein_id)
print("Label:", int(labels.detach().cpu().view(-1)[0].item()))

drug_rep = torch.load(f"drug/embeddings_atomic/{drug_id[0]}_unimol.pt", weights_only=False)
print(np.array(drug_rep["atomic_reprs"]))  # (N_atoms, 512)
print(np.array(drug_rep["atomic_reprs"]).shape)  # (N_atoms, 512)
print(np.array(drug_rep["atomic_symbol"]))  # (N_atoms, 512)
print(np.array(drug_rep["atomic_symbol"]).shape)  # (N_atoms, 512)
print(np.array(drug_rep["atomic_coords"]))  # (N_atoms, 512)
print(np.array(drug_rep["atomic_coords"]).shape)  # (N_atoms, 512)

model.eval()
with torch.no_grad():
    predictions, attentionp, attentiond = model(
        protein_emb,
        drug_emb,
        protein_mask=protein_mask,
        drug_mask=drug_mask,
        return_attention=True,
    )

print("Predictions:", predictions, predictions.shape)
print("Protein Attention:", attentionp, attentionp.shape)
print("Drug Attention:", attentiond, attentiond.shape)

# ---- attention → numpy (batch 0) ----
# For nn.MultiheadAttention with batch_first=True and default average_attn_weights=True,
# attention shapes are (B, L_target, L_source) with no explicit head dimension.
# So we just take the first batch element directly.
# attentionp: (B, Lp, Ld)  protein queries → drug keys
# attentiond: (B, Ld, Lp)  drug queries → protein keys
attn_p = attentionp[0].detach().cpu().numpy()  # (Lp, Ld)
attn_d = attentiond[0].detach().cpu().numpy()  # (Ld, Lp)

# ---- build atom labels for drug dimension ----
# drug_rep['atomic_symbol'] is expected to contain one symbol per atom (possibly with a SEP at the end)
atom_labels = np.array(drug_rep["atomic_symbol"]).reshape(-1)

# align labels with number of drug tokens (Ld)
Ld = attn_p.shape[1]
if atom_labels.shape[0] == Ld + 1:
    # drop SEP or extra token
    atom_labels = atom_labels[:-1]
elif atom_labels.shape[0] > Ld:
    atom_labels = atom_labels[:Ld]
elif atom_labels.shape[0] < Ld:
    # pad with generic labels if needed
    pad_len = Ld - atom_labels.shape[0]
    atom_labels = np.concatenate([atom_labels, np.array([f"X{i}" for i in range(pad_len)])])

# choose an atom index to highlight (0-based). Change this depending on which atom you care about.
highlight_atom_idx = 0
highlight_atom_idx = max(0, min(highlight_atom_idx, Ld - 1))
# (later we may trim SEP for rendering; keep highlight index within non-SEP atoms)

# ---- symmetric interaction map (residue ↔ atom) ----
# attn_p: (Lp, Ld) protein queries → drug tokens
# attn_d: (Ld, Lp) drug queries → protein residues
# We define a symmetric score I(i,j) combining both directions.
# Use arithmetic mean here; you can switch to geometric mean if desired.
inter_map = np.sqrt(attn_p * attn_d.T + 1e-12)  # (Lp, Ld)

# ---- heatmap: protein residues (rows) vs drug atoms (cols) ----
plt.figure(figsize=(8, 6))
plt.imshow(attn_p, aspect="auto", interpolation="nearest")
plt.colorbar(label="Attention weight")
plt.xlabel("Drug atoms")
plt.ylabel("Protein residues")
plt.title(f"{protein_id} Protein queries → drug atoms {drug_id}")

# x-axis: label each column by atom symbol
x_positions = np.arange(attn_p.shape[1])
plt.xticks(x_positions, atom_labels, rotation=90, fontsize=6)

# highlight chosen atom column
plt.axvline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

# ---- heatmap: drug atoms (rows) vs protein residues (cols) ----
plt.figure(figsize=(8, 6))
plt.imshow(attn_d, aspect="auto", interpolation="nearest")
plt.colorbar(label="Attention weight")
plt.xlabel("Protein residues")
plt.ylabel("Drug atoms")
plt.title(f"{drug_id} Drug queries → protein residues {protein_id}")

# y-axis: label each row by atom symbol
y_positions = np.arange(attn_d.shape[0])
plt.yticks(y_positions, atom_labels, fontsize=6)

# highlight chosen atom row
plt.axhline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

# ---- heatmap: symmetric interaction map (residue ↔ atom) ----
plt.figure(figsize=(8, 6))
plt.imshow(inter_map, aspect="auto", interpolation="nearest")
plt.colorbar(label="Symmetric attention score")
plt.xlabel("Drug atoms")
plt.ylabel("Protein residues")
plt.title("Symmetric residue–atom interaction map")

x_positions = np.arange(inter_map.shape[1])
plt.xticks(x_positions, atom_labels, rotation=90, fontsize=6)

# highlight chosen atom column
plt.axvline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

plt.show()

#TODO: pre extract protein chains so i can use them for this purpose


# ---- 3D render drug atoms with py3Dmol (XYZ) ----
view = py3Dmol.view(width=800, height=800)

# coords should be (N, 3), symbols should be (N,)
atomic_coords = np.array(drug_rep["atomic_coords"], dtype=float)

# Uni-Mol sometimes stores coords with a leading batch dimension, e.g. (1, N, 3)
# or other extra singleton dims. Squeeze them safely.
if atomic_coords.ndim >= 3:
    atomic_coords = np.squeeze(atomic_coords)

# After squeeze, ensure we have either (N, 3) or something we can reduce to it.
# If coords are (N, 3, 3) (e.g., multiple atom reference points), take the central atom (index 1).
if atomic_coords.ndim == 3 and atomic_coords.shape[-2:] == (3, 3):
    atomic_coords = atomic_coords[:, 1, :]

# Symbols sometimes come as bytes / numpy scalars; normalize to clean strings early
atomic_symbols_raw = np.array(drug_rep["atomic_symbol"], dtype=object).reshape(-1)
atomic_symbols = np.array([str(s).strip() for s in atomic_symbols_raw], dtype=object)

# If coords come with an extra trailing dimension or a SEP token, trim consistently.
# Common Uni-Mol artifacts: an extra "SEP" symbol or an extra coord row.
N = min(len(atomic_symbols), atomic_coords.shape[0])
atomic_symbols = atomic_symbols[:N]
atomic_coords = atomic_coords[:N]

# Drop SEP-like tokens (Uni-Mol variants): 'SEP', '[SEP]', '<SEP>'
atomic_coords = atomic_coords[:-1]
atomic_symbols = atomic_symbols[:-1]
# Align per-atom weights with the rendered atoms.
# We use the symmetric residue–atom interaction map column-sum as an atom importance score.
# inter_map is (Lp, Ld) and its columns correspond to the same drug-token axis as atom_labels.
atom_weights = inter_map.sum(axis=0)  # (Ld,)
atom_weights = atom_weights[:len(atomic_symbols)]

# Ensure coords are (N, 3)
if atomic_coords.ndim != 2 or atomic_coords.shape[1] != 3:
    raise ValueError(
        "atomic_coords must be shape (N,3) after squeezing; "
        f"got {atomic_coords.shape}. "
        "If this is (N,3,3), we try to reduce it above; otherwise inspect drug_rep['atomic_coords']."
    )

# Build a proper XYZ string: first line = atom count, second = comment
xyz_lines = [str(len(atomic_symbols)), f"{drug_id[0]} atoms"]
for sym, (x, y, z) in zip(atomic_symbols, atomic_coords):
    xyz_lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
xyz_str = "\n".join(xyz_lines) + "\n"

print(xyz_str)

view.addModel(xyz_str, "xyz")

#
# Base style: spheres for all atoms (optionally keep sticks too)
SPHERE_RADIUS = 0.20  # constant radius for all atoms; tweak to taste
view.setStyle({"sphere": {"radius": SPHERE_RADIUS}, "stick": {"radius": 0.08}})

# Per-atom styling by weight: larger & more saturated for higher weights
# Normalize weights safely
w = np.asarray(atom_weights, dtype=float)
if w.size == 0:
    w = np.ones(len(atomic_symbols), dtype=float)

w_min, w_max = float(np.min(w)), float(np.max(w))
if abs(w_max - w_min) < 1e-12:
    w_norm = np.zeros_like(w)
else:
    w_norm = (w - w_min) / (w_max - w_min)

# Map to colors using a matplotlib colormap (no seaborn)
import matplotlib
cmap = matplotlib.cm.get_cmap("viridis")

def rgba_to_hex(rgba):
    r, g, b, _ = rgba
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

# Apply sphere overlay per atom (py3Dmol uses 1-based serial indices)
for i, wn in enumerate(w_norm):
    # Color only: keep a constant sphere radius for every atom
    color = rgba_to_hex(cmap(float(wn)))
    view.addStyle({"serial": i + 1}, {"sphere": {"radius": SPHERE_RADIUS, "color": color}})

# Optional: emphasize one chosen atom on top of the weight styling
# view.addStyle({"serial": highlight_atom_idx + 1}, {"sphere": {"radius": 0.85, "color": "#ff2d2d"}})

view.zoomTo()

# In a Jupyter notebook, view.show() will display.
# In a plain Python run, write an HTML file you can open in a browser.

from pathlib import Path
html_path = Path("drug_view.html").resolve()
html_path.write_text(view._make_html(), encoding="utf-8")
print(f"Wrote 3D view to: {html_path}")