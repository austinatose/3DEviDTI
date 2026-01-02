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
parser.add_argument("--ckpt", type=str, default="saved/cnn321_epoch_58.pt", help="Path to checkpoint")
parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda/mps")
parser.add_argument("--csv", type=str, default="lists/db_train.csv", help="CSV file to search")
parser.add_argument("--drug_id", type=str, default=None, help="DrugBank ID to find (e.g., DB00001)")
parser.add_argument("--uniprot_id", type=str, default=None, help="UniProt ID to find (e.g., P00533)")
parser.add_argument("--max_search", type=int, default=20000, help="Max batches to scan for the pair")
parser.add_argument("--max_matches", type=int, default=20, help="Max number of matches to collect and list when searching")
parser.add_argument(
    "--pick_match",
    type=int,
    default=None,
    help=(
        "When multiple matches are found, choose which match to inspect (0-based index into the printed list). "
        "If omitted and multiple matches exist, the script will list matches and exit."
    ),
)
parser.add_argument("--top_k", type=int, default=20, help="How many top-weighted indices to print for protein/drug")
parser.add_argument(
    "--atom_labels",
    type=str,
    default="topk",
    choices=["none", "topk", "all"],
    help="Add atom labels in the py3Dmol 3D view: none/topk/all. topk labels atoms in --top_k by attention.",
)
parser.add_argument(
    "--label_heavy_only",
    action="store_true",
    help="If set, only label heavy atoms (non-H) in the py3Dmol view.",
)
parser.add_argument(
    "--label_font",
    type=int,
    default=11,
    help="Font size for py3Dmol labels.",
)
parser.add_argument(
    "--label_with_weight",
    action="store_true",
    help="If set, include attention weight in the atom label text.",
)
parser.add_argument(
    "--joint_mode",
    type=str,
    default="prod",
    choices=["geom", "arith", "harm", "prod", "min"],
    help=(
        "How to combine the two directional attentions into a joint residue↔atom score. "
        "geom=sqrt(p*d) (agreement, scale-stable), arith=(p+d)/2, harm=2pd/(p+d), "
        "prod=p*d (strong agreement, very peaky), min=min(p,d) (strict agreement)."
    ),
)
parser.add_argument(
    "--label",
    type=str,
    default=None,
    help="Filter by label: 'pos'/'positive'/'1' for positive, 'neg'/'negative'/'0' for negative. If omitted, no label filter.",
)
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
# Use deterministic ordering when searching for a specific pair, so listed matches are stable.
shuffle_flag = False if (args.drug_id is not None or args.uniprot_id is not None) else True

test_dl = DataLoader(test_ds, batch_size=1, shuffle=shuffle_flag, num_workers=0, collate_fn=collate_fn, drop_last=False)


# ---- label filter helper ----
def _parse_label_filter(x: str):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "pos", "positive", "+"}:
        return 1
    if s in {"0", "neg", "negative", "-"}:
        return 0
    raise ValueError(f"Unrecognized --label={x!r}. Use pos/neg or 1/0.")


def _batch_label_int(batch) -> int:
    """Robustly extract a 0/1 label from a batch."""
    y = batch.get("label", None)
    if y is None:
        return None
    # Most likely a tensor of shape (B,) or (B,1)
    try:
        return int(y.detach().cpu().view(-1)[0].item())
    except Exception:
        pass
    # Sometimes label is a list/tuple
    try:
        return int(y[0])
    except Exception:
        return None


label_filter = _parse_label_filter(args.label)
if label_filter is not None:
    print(f"Label filter enabled: {label_filter} ({'positive' if label_filter==1 else 'negative'})")

# ---- find a specific (drug_id, uniprot_id) pair if requested ----
selected = None
if args.drug_id is not None or args.uniprot_id is not None:
    print(f"Searching for pair: drug_id={args.drug_id}, uniprot_id={args.uniprot_id} in {args.csv}")

    matches = []  # list of (batch_index, drug_id, uniprot_id, label, batch)
    for i, batch in enumerate(test_dl):
        if i >= args.max_search:
            break

        b_drug = batch.get("drug_id", [None])[0]
        b_prot = batch.get("uniprot_id", [None])[0]

        ok_drug = (args.drug_id is None) or (b_drug == args.drug_id)
        ok_prot = (args.uniprot_id is None) or (b_prot == args.uniprot_id)

        b_label = _batch_label_int(batch)
        ok_label = (label_filter is None) or (b_label == label_filter)

        if ok_drug and ok_prot and ok_label:
            matches.append((i, b_drug, b_prot, b_label, batch))
            if len(matches) >= int(args.max_matches):
                break

    if len(matches) == 0:
        raise RuntimeError(
            "Pair not found. Check IDs, or increase --max_search, or confirm the CSV contains that pair. "
            f"Requested drug_id={args.drug_id}, uniprot_id={args.uniprot_id}."
        )

    # Always list matches if we have more than one
    if len(matches) > 1:
        print(f"\nFound {len(matches)} matches (showing up to --max_matches={args.max_matches}):")
        for k, (i, b_drug, b_prot, b_label, _) in enumerate(matches):
            print(f"  [{k}] batch={i}  drug_id={b_drug}  uniprot_id={b_prot}  label={b_label}")

        if args.pick_match is None:
            raise SystemExit(
                "\nMultiple matches found. Re-run with --pick_match <k> to choose one (e.g., --pick_match 0)."
            )

        pick = int(args.pick_match)
        if pick < 0 or pick >= len(matches):
            raise SystemExit(f"--pick_match {pick} out of range for {len(matches)} matches")

        selected = matches[pick][4]
        print(f"\nSelected match [{pick}] from batch={matches[pick][0]}")

    else:
        # single match
        selected = matches[0][4]
        print(f"Found at batch {matches[0][0]}: drug_id={matches[0][1]}, uniprot_id={matches[0][2]}, label={matches[0][3]}")

else:
    # fallback: label-aware sampling
    if label_filter is None:
        selected = next(iter(test_dl))
    else:
        selected = None
        for i, batch in enumerate(test_dl):
            if i >= args.max_search:
                break
            b_label = _batch_label_int(batch)
            if b_label == label_filter:
                selected = batch
                print(f"Selected first matching label at batch {i}: label={b_label}")
                break
        if selected is None:
            raise RuntimeError(
                f"No sample with label={label_filter} found within --max_search={args.max_search} batches in {args.csv}."
            )

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

# ---- joint / symmetric attention map (protein residues ↔ drug atoms) ----
# attn_p[i,j] : P(drug=j | prot_query=i)
# attn_d[j,i] : P(prot=i | drug_query=j)
# A good "joint" score should reward *agreement* between both directions.
# We combine them elementwise with several options.

def compute_joint_map(attn_p: np.ndarray, attn_d: np.ndarray, mode: str = "geom", eps: float = 1e-12) -> np.ndarray:
    # attn_d is (Ld, Lp); transpose to align with attn_p (Lp, Ld)
    d = attn_d.T
    p = attn_p
    if p.shape != d.shape:
        raise ValueError(f"Shape mismatch: attn_p {p.shape} vs attn_d.T {d.shape}")

    if mode == "geom":
        # geometric mean: stable, highlights mutual agreement without being overly peaky
        return np.sqrt(np.maximum(p, 0.0) * np.maximum(d, 0.0) + eps)
    if mode == "arith":
        # arithmetic mean: more forgiving (can highlight one-directional mass)
        return 0.5 * (p + d)
    if mode == "harm":
        # harmonic mean: punishes disagreement more than arithmetic
        return (2.0 * p * d) / (p + d + eps)
    if mode == "prod":
        # raw product: strongest agreement but can become extremely peaky
        return p * d
    if mode == "min":
        # strict agreement: limited by the weaker direction
        return np.minimum(p, d)

    raise ValueError(f"Unknown joint mode: {mode}")


joint_map = compute_joint_map(attn_p, attn_d, mode=args.joint_mode)

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

# ---- top-k weighted indices (protein + drug), using *individual* attention maps ----
# Individual-direction marginals (kept for reference/debugging)
protein_weights = attn_d.sum(axis=0)  # (Lp,)  drug→protein mass
drug_weights = attn_p.sum(axis=0)     # (Ld,)  protein→drug mass

# Joint/symmetric marginal mass (used for ranking/plots)
# joint_map is (Lp, Ld): residues x atoms
protein_joint_weights = joint_map.sum(axis=1)  # (Lp,)
drug_joint_weights = joint_map.sum(axis=0)     # (Ld,)

# Use joint mass for downstream visualization/ranking by default
atom_mass_for_render = drug_joint_weights

# Use masks (if present) to ignore padding positions.
# NOTE: Different codebases use different conventions:
#   - "valid mask" (1/True = real token)
#   - "padding mask" / key_padding_mask (1/True = PAD token)
# We'll auto-detect by choosing the interpretation that yields the most valid tokens.

def _infer_valid(mask_tensor, L: int, name: str):
    try:
        m = mask_tensor[0].detach().cpu().numpy().reshape(-1)
        if m.shape[0] != L:
            raise ValueError(f"{name} mask length {m.shape[0]} != L {L}")
        # Candidate A: treat nonzero/True as valid
        cand_a = m.astype(bool)
        # Candidate B: treat nonzero/True as padding (so valid is invert)
        cand_b = ~cand_a
        # Choose whichever gives more valid tokens (and at least 1)
        a_sum = int(np.sum(cand_a))
        b_sum = int(np.sum(cand_b))
        valid = cand_a if a_sum >= b_sum else cand_b
        # Fallback: if everything is invalid, keep all valid so we can still debug
        if int(np.sum(valid)) == 0:
            valid = np.ones(L, dtype=bool)
        # Debug print to catch convention mistakes quickly
        print(f"[mask] {name}: len={L}  cand_valid_sum={a_sum}  cand_pad_sum={b_sum}  using={'A(valid=mask)' if a_sum>=b_sum else 'B(valid=~mask)'}")
        return valid
    except Exception as e:
        print(f"[mask] {name}: could not infer ({e}); defaulting to all-valid")
        return np.ones(L, dtype=bool)

prot_valid = _infer_valid(protein_mask, protein_weights.shape[0], "protein")
drug_valid = _infer_valid(drug_mask, drug_weights.shape[0], "drug")

# Apply validity by setting invalid weights to -inf so they won't appear in top-k
# Protein ranking uses JOINT marginal mass
protein_weights_f = protein_joint_weights.copy()
protein_weights_f[~prot_valid] = -np.inf

# Drug ranking uses JOINT marginal mass
drug_weights_f = drug_joint_weights.copy()
drug_weights_f[~drug_valid] = -np.inf

k_prot = max(1, min(int(args.top_k), int(np.sum(prot_valid))))
k_drug = max(1, min(int(args.top_k), int(np.sum(drug_valid))))


prot_top_idx = np.argsort(-protein_weights_f)[:k_prot]
drug_top_idx = np.argsort(-drug_weights_f)[:k_drug]

print(f"\n==== Top protein indices by JOINT residue↔atom mass ({args.joint_mode}) ====")
for rank, idx in enumerate(prot_top_idx, 1):
    w_i = float(protein_joint_weights[idx])
    # Also print 1-based index for convenience
    print(f"{rank:>2}. prot_idx={idx} (1-based {idx+1})  weight={w_i:.6g}")

print(f"\n==== Top drug indices by JOINT residue↔atom mass ({args.joint_mode}) ====")
for rank, jdx in enumerate(drug_top_idx, 1):
    w_j = float(drug_joint_weights[jdx])
    # atom_labels is aligned to the Ld axis above (may be padded/truncated)
    lbl = atom_labels[jdx] if jdx < len(atom_labels) else f"atom{jdx}"
    print(f"{rank:>2}. drug_idx={jdx} (1-based {jdx+1})  atom={lbl}  weight={w_j:.6g}")
print("====\n")

# ---- 1D bar plots: per-position mass from the JOINT (symmetric) residue↔atom map ----
# joint_map is (Lp, Ld) aligned to (protein residues, drug atoms).
# We plot the marginal sums of the joint map:
#   - per-residue joint mass: sum over atoms
#   - per-atom joint mass:    sum over residues

protein_joint_weights = joint_map.sum(axis=1)  # (Lp,)
drug_joint_weights = joint_map.sum(axis=0)     # (Ld,)

# Apply validity masks (padding/non-residue tokens)
protein_weights_plot = protein_joint_weights.copy()
protein_weights_plot[~prot_valid] = 0.0

drug_weights_plot = drug_joint_weights.copy()
drug_weights_plot[~drug_valid] = 0.0

# Protein bar plot
plt.figure(figsize=(12, 3))
xp = np.arange(len(protein_weights_plot))
plt.bar(xp, protein_weights_plot)
plt.xlabel("Protein residue index")
plt.ylabel(f"Joint mass ({args.joint_mode})")
plt.title(f"Protein per-residue joint mass  {protein_id} ↔ {drug_id}  ({args.joint_mode})")

# thin ticks
prot_stride = max(1, int(len(xp) // 50))
prot_ticks = xp[::prot_stride]
plt.xticks(prot_ticks, [str(int(t)) for t in prot_ticks], rotation=0, fontsize=6)
plt.tight_layout()

# Drug bar plot
plt.figure(figsize=(12, 3))
xd = np.arange(len(drug_weights_plot))
plt.bar(xd, drug_weights_plot)
plt.xlabel("Drug atom index")
plt.ylabel(f"Joint mass ({args.joint_mode})")
plt.title(f"Drug per-atom joint mass  {protein_id} ↔ {drug_id}  ({args.joint_mode})")

# thin ticks + use atom symbols when possible
atom_labels_safe = np.array([str(a) for a in atom_labels], dtype=object)
drug_stride = max(1, int(len(xd) // 50))
drug_ticks = xd[::drug_stride]
plt.xticks(
    drug_ticks,
    [atom_labels_safe[int(t)] if int(t) < len(atom_labels_safe) else str(int(t)) for t in drug_ticks],
    rotation=90,
    fontsize=6,
)
plt.tight_layout()

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
# plt.axvline(highlight_atom_idx, linestyle="--")

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
# plt.axhline(highlight_atom_idx, linestyle="--")

plt.tight_layout()

# ---- heatmap: joint residue↔atom score (rows=residues, cols=atoms) ----
plt.figure(figsize=(8, 6))
plt.imshow(joint_map, aspect="auto", interpolation="nearest")
plt.colorbar(label=f"Joint score ({args.joint_mode})")
plt.xlabel("Drug atoms")
plt.ylabel("Protein residues")
plt.title(f"Joint residue↔atom map ({args.joint_mode})  {protein_id} ↔ {drug_id}")

x_positions = np.arange(joint_map.shape[1])
plt.xticks(x_positions, atom_labels, rotation=90, fontsize=6)

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

# Align the rendered atoms with the same indexing used by attention.
# IMPORTANT: Do NOT blindly drop the last atom; instead detect and remove special tokens
# like SEP/CLS consistently from symbols/coords/weights, and build an index mapping.

# First, clamp everything to a common original length.
N0 = min(len(atomic_symbols), atomic_coords.shape[0], int(drug_weights.shape[0]))
atomic_symbols = atomic_symbols[:N0]
atomic_coords = atomic_coords[:N0]
atom_weights_full = atom_mass_for_render[:N0].copy()  # weights aligned to the original token axis

SPECIAL = {"SEP", "[SEP]", "<SEP>", "CLS", "[CLS]", "<CLS>", "BOS", "[BOS]", "<BOS>", "EOS", "[EOS]", "<EOS>"}

# Identify special tokens by symbol string and also by obviously-invalid coordinates.
# (Some pipelines encode SEP as a dummy row of zeros.)
syms_upper = np.array([str(s).strip().upper() for s in atomic_symbols], dtype=object)
is_special_sym = np.array([u in SPECIAL or u == "" for u in syms_upper], dtype=bool)

coords = np.asarray(atomic_coords, dtype=float)
is_special_coord = np.zeros(coords.shape[0], dtype=bool)
try:
    is_special_coord = np.all(np.isfinite(coords), axis=1) & (np.linalg.norm(coords, axis=1) < 1e-6)
except Exception:
    pass

special_mask = is_special_sym | is_special_coord

# Build mapping from original token index -> rendered atom index
orig_to_render = -np.ones(N0, dtype=int)
keep_idx = []
for i in range(N0):
    if not bool(special_mask[i]):
        orig_to_render[i] = len(keep_idx)
        keep_idx.append(i)

keep_idx = np.array(keep_idx, dtype=int)

if keep_idx.size == 0:
    raise ValueError("After removing special tokens, no atoms remain to render.")

# Apply the same filtering to symbols/coords/weights so serial indices match label/color
atomic_symbols = atomic_symbols[keep_idx]
atomic_coords = atomic_coords[keep_idx]
atom_weights = atom_weights_full[keep_idx]

print(f"[render] N0={N0}  kept={len(keep_idx)}  removed_special={int(np.sum(special_mask))}")

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

# ---- optional atom labels (serial is 1-based in 3Dmol.js) ----
# NOTE: With XYZ, we only have element symbols + coordinates (no chemical atom names).
# We therefore label by element + index (and optionally weight), which is still enough
# to cross-reference your `drug_idx` indices.
if args.atom_labels != "none":
    try:
        view.removeAllLabels()
    except Exception:
        pass

    if args.atom_labels == "all":
        label_indices = list(range(len(atomic_symbols)))
    else:
        # drug_top_idx is in the original attention token indexing; map to rendered indices
        label_indices = []
        for orig_i in drug_top_idx:
            oi = int(orig_i)
            if 0 <= oi < orig_to_render.shape[0]:
                ri = int(orig_to_render[oi])
                if ri >= 0 and ri < len(atomic_symbols):
                    label_indices.append(ri)

    for i in label_indices:
        sym = str(atomic_symbols[i])
        if args.label_heavy_only and sym.upper() == "H":
            continue

        if args.label_with_weight and i < len(atom_weights):
            txt = f"{sym}{i}  w={float(atom_weights[i]):.3g}"
        else:
            txt = f"{sym}{i}"

        view.addLabel(
            txt,
            {
                "fontSize": int(args.label_font),
                "backgroundColor": "white",
                "fontColor": "black",
                "inFront": True,
            },
            {"serial": int(i) + 1},
        )

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