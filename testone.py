import argparse
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from config.cfg import get_cfg_defaults
from model import Model


# Keys are 1-based UniProt FASTA indices for P35354 (the 17-residue signal peptide
# is included in the FASTA, so PDB-mature numbering = FASTA index - 14 here, not +14).
# Canonical-anchor labels in comments use the conventional PDB-mature numbering.
PLIP_CONTACTS = {
    106: ['halogen'],
    189: ['hbond'],          # Gln192 (PDB-mature), canonical sulfonamide anchor
    335: ['hydrophobic'],
    338: ['hydrophobic'],    # Leu352 (PDB-mature), canonical
    339: ['hbond'],
    341: ['hbond'],
    370: ['hydrophobic'],
    371: ['hydrophobic'],    # Tyr385 (PDB-mature), canonical
    373: ['hydrophobic'],    # Trp387 (PDB-mature), canonical
    499: ['hbond'],          # Arg513 (PDB-mature), canonical
    504: ['hbond', 'hydrophobic'],
    509: ['hydrophobic'],
    513: ['hydrophobic'],
}

GLN192_FASTA_KEY = 189  # PDB-mature 192 -> UniProt FASTA 189; chart label stays "Gln192"

# FASTA, PLIP, and the embeddings have inconsistent numbering schemes.
# FASTA alignment was previously performed. The PLIP indices here are separately aligned directly to the embeddings

def _load_protein_embedding(path: str) -> torch.Tensor:
	if os.path.isdir(path):
		files = sorted(glob.glob(os.path.join(path, "*.pt")))
		if not files:
			raise FileNotFoundError(f"No .pt files found in directory: {path}")
		path = files[-1]

	emb = torch.load(path, map_location="cpu", weights_only=True)
	if not isinstance(emb, torch.Tensor):
		emb = torch.as_tensor(emb)
	emb = emb.to(dtype=torch.float32, copy=False).contiguous()
	if emb.ndim != 2:
		raise ValueError(f"Protein embedding must be 2D; got shape {tuple(emb.shape)}")
	return emb


def _load_drug_embedding(path: str) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
	obj = torch.load(path, map_location="cpu", weights_only=False)
	if isinstance(obj, dict) and "atomic_reprs" in obj:
		emb = torch.from_numpy(np.asarray(obj["atomic_reprs"], dtype=np.float32)).reshape(-1, 512)
		labels = None
		if "atomic_symbol" in obj:
			labels = np.array(obj["atomic_symbol"], dtype=object).reshape(-1)
		return emb, labels

	emb = torch.as_tensor(obj, dtype=torch.float32)
	if emb.ndim != 2:
		raise ValueError(f"Drug embedding must be 2D; got shape {tuple(emb.shape)}")
	return emb, None


def _trim_sep(emb: torch.Tensor) -> torch.Tensor:
    return emb[:-1, :] if emb.size(0) > 1 else emb


def compute_joint_map(attn_p: np.ndarray, attn_d: np.ndarray, mode: str = "geom", eps: float = 1e-12) -> np.ndarray:
	d = attn_d.T
	p = attn_p
	if p.shape != d.shape:
		raise ValueError(f"Shape mismatch: attn_p {p.shape} vs attn_d.T {d.shape}")

	if mode == "geom":
		return np.sqrt(np.maximum(p, 0.0) * np.maximum(d, 0.0) + eps)
	if mode == "arith":
		return 0.5 * (p + d)
	if mode == "harm":
		return (2.0 * p * d) / (p + d + eps)
	if mode == "prod":
		return p * d
	if mode == "min":
		return np.minimum(p, d)

	raise ValueError(f"Unknown joint mode: {mode}")


def _align_atom_labels(labels: Optional[np.ndarray], Ld: int) -> Optional[np.ndarray]:
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=object).reshape(-1)
    if labels.shape[0] == Ld + 1:
        labels = labels[:-1]
    if labels.shape[0] < Ld:
        pad = np.array([f"X{i}" for i in range(Ld - labels.shape[0])], dtype=object)
        labels = np.concatenate([labels, pad])
    return labels[:Ld]


def _resolve_map_json(protein_emb_path: str) -> Optional[str]:
    """Mirror biotite_mapping output layout: interpretation/mappings/<protein_id>/<stem>.map.json."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(protein_emb_path)))
    stem = os.path.basename(protein_emb_path).split(".")[0]  # strip every dotted suffix (e.g. ".cif.pt")
    candidate = os.path.join("interpretation", "mappings", parent, f"{stem}.map.json")
    return candidate if os.path.exists(candidate) else None


def _load_fasta_to_struct(map_json_path: str) -> List[Optional[int]]:
    with open(map_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    arr = data.get("fasta_to_struct")
    if not isinstance(arr, list):
        raise ValueError(f"map_json missing fasta_to_struct: {map_json_path}")
    return [None if x is None else int(x) for x in arr]


def _apply_sparse_ticks(ax, labels: np.ndarray, axis: str, max_ticks: int = 50) -> None:
    stride = max(1, int(len(labels) // max_ticks))
    ticks = np.arange(0, len(labels), stride)
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(labels[i]) for i in ticks], rotation=0, fontsize=7)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(labels[i]) for i in ticks], fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--protein_emb", type=str, required=True)
    parser.add_argument("--drug_emb", type=str, required=True)
    parser.add_argument("--protein_id", type=str, default=None)
    parser.add_argument("--drug_id", type=str, default=None)
    parser.add_argument(
        "--joint_mode",
        type=str,
        default="prod",
        choices=["geom", "arith", "harm", "prod", "min"],
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--map_json",
        type=str,
        default=None,
        help="Path to a .map.json (FASTA<->struct). If omitted, auto-resolve from --protein_emb. "
             "Used to align PLIP FASTA indices onto the model's struct-indexed chart.",
    )
    args = parser.parse_args()

    map_json_path = args.map_json or _resolve_map_json(args.protein_emb)
    fasta_to_struct: Optional[List[Optional[int]]] = None
    if map_json_path is not None:
        try:
            fasta_to_struct = _load_fasta_to_struct(map_json_path)
        except Exception as exc:
            print(f"[warn] failed to load mapping {map_json_path}: {exc}")
            fasta_to_struct = None

    cfg = get_cfg_defaults()
    device = torch.device(args.device)

    protein_emb = _trim_sep(_load_protein_embedding(args.protein_emb))
    drug_emb, atom_labels = _load_drug_embedding(args.drug_emb)
    drug_emb = _trim_sep(drug_emb)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Model(cfg=cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    p_mask = torch.zeros(protein_emb.shape[0], dtype=torch.bool)
    d_mask = torch.zeros(drug_emb.shape[0], dtype=torch.bool)

    p_emb = protein_emb.unsqueeze(0)
    d_emb = drug_emb.unsqueeze(0)
    p_mask = p_mask.unsqueeze(0)
    d_mask = d_mask.unsqueeze(0)

    with torch.no_grad():
        logits, attentionp, attentiond = model(
            p_emb,
            d_emb,
            protein_mask=p_mask,
            drug_mask=d_mask,
            return_attention=True,
        )

    print(logits)

    attn_p = attentionp[0].detach().cpu().numpy()
    attn_d = attentiond[0].detach().cpu().numpy()

    Ld = attn_p.shape[1]
    atom_labels = _align_atom_labels(atom_labels, Ld)

    joint_map = compute_joint_map(attn_p, attn_d, mode=args.joint_mode)
    protein_joint = joint_map.sum(axis=1)
    drug_joint = joint_map.sum(axis=0)

    pair_label = " / ".join([
        args.protein_id or os.path.basename(args.protein_emb),
        args.drug_id or os.path.basename(args.drug_emb),
    ])

    import matplotlib.pyplot as plt

    # Protein-to-drug heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(attn_p, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(img, ax=ax, pad=0.01, fraction=0.03, shrink=0.9)
    cb.set_label("Attention weight")
    ax.set_xlabel("Drug atoms")
    ax.set_ylabel("Protein residues")
    ax.set_title(f"Protein queries to drug atoms - {pair_label}")
    if atom_labels is not None:
        _apply_sparse_ticks(ax, atom_labels, axis="x")
    fig.tight_layout()

    # Drug-to-protein heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(attn_d, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(img, ax=ax, pad=0.01, fraction=0.03, shrink=0.9)
    cb.set_label("Attention weight")
    ax.set_xlabel("Protein residues")
    ax.set_ylabel("Drug atoms")
    ax.set_title(f"Drug queries to protein residues - {pair_label}")
    if atom_labels is not None:
        _apply_sparse_ticks(ax, atom_labels, axis="y")
    fig.tight_layout()

    # Joint heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(joint_map, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(img, ax=ax, pad=0.01, fraction=0.03, shrink=0.9)
    cb.set_label(f"Joint score ({args.joint_mode})")
    ax.set_xlabel("Drug atoms")
    ax.set_ylabel("Protein residues")
    ax.set_title(f"Joint residue-atom map ({args.joint_mode}) - {pair_label}")
    if atom_labels is not None:
        _apply_sparse_ticks(ax, atom_labels, axis="x")
    fig.tight_layout()

    # Protein bar chart
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(protein_joint)), protein_joint)
    ax.set_xlabel("Protein residue index")
    ax.set_ylabel(f"Joint mass ({args.joint_mode})")
    ax.set_title(f"Protein per-residue joint mass - {pair_label}")

    from matplotlib.lines import Line2D

    plip_color = "C3"
    shape_for = {"hbond": "o", "hydrophobic": "s", "halogen": "^"}
    ymax = float(np.nanmax(protein_joint)) if protein_joint.size else 0.0

    def _fasta1_to_struct0(fasta_1based: int) -> Optional[int]:
        """Translate a 1-based FASTA index to a 0-based struct (bar-chart) index."""
        if fasta_to_struct is None:
            return None
        fasta_0 = fasta_1based - 1
        if fasta_0 < 0 or fasta_0 >= len(fasta_to_struct):
            return None
        s = fasta_to_struct[fasta_0]
        if s is None or s < 0 or s >= len(protein_joint):
            return None
        return s

    plip_xs = {
        residue: _fasta1_to_struct0(residue) for residue in PLIP_CONTACTS
    }
    have_overlay = ymax > 0.0 and fasta_to_struct is not None and any(
        v is not None for v in plip_xs.values()
    )

    if have_overlay:
        # Flatten to one (x, residue, type) per glyph, then greedy-cluster left-to-right
        # so markers whose x positions are too close to render side-by-side stack vertically.
        markers: list[tuple[int, int, str]] = []
        skipped: list[int] = []
        for residue_idx, types in PLIP_CONTACTS.items():
            x = plip_xs.get(residue_idx)
            if x is None:
                skipped.append(residue_idx)
                continue
            for t in types:
                markers.append((x, residue_idx, t))
        markers.sort(key=lambda m: (m[0], m[1]))

        # cluster_gap relative to axis range (marker s=55 -> ~7pt diameter,
        # ~7 struct units on a 10-inch axis for L_struct ~ 550).
        cluster_gap = max(3.0, len(protein_joint) * 0.012)
        rows: list[int] = []
        cluster_start = 0
        prev_x: Optional[int] = None
        for i, (x, _, _) in enumerate(markers):
            if prev_x is None or (x - prev_x) > cluster_gap:
                cluster_start = i
                rows.append(0)
            else:
                rows.append(i - cluster_start)
            prev_x = x
        max_row = max(rows) if rows else 0

        band_base = ymax * 1.04
        row_offset = ymax * 0.04
        y_upper = max(ymax * 1.15, band_base + (max_row + 1) * row_offset)
        ax.set_ylim(0, y_upper)

        for (x, _residue, t), row in zip(markers, rows):
            marker_y = band_base + row * row_offset
            curve_y = float(protein_joint[x])
            ax.plot(
                [x, x], [curve_y, marker_y],
                linestyle=":", linewidth=1.0, color=plip_color, alpha=0.25, zorder=9,
            )
            ax.scatter(
                [x], [marker_y],
                marker=shape_for[t], s=55, linewidths=1.5,
                facecolor="none", edgecolor=plip_color, zorder=10,
            )

        gln192_x = plip_xs.get(GLN192_FASTA_KEY)
        if gln192_x is not None:
            gln_y = band_base
            for (x, residue, _t), row in zip(markers, rows):
                if residue == GLN192_FASTA_KEY:
                    gln_y = band_base + row * row_offset
                    break
            ax.annotate(
                "Gln192",
                xy=(gln192_x, gln_y),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8, color=plip_color, va="center",
            )

        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=plip_color, markersize=7, markeredgewidth=1.5,
                   label="PLIP H-bond"),
            Line2D([0], [0], marker="s", linestyle="", markerfacecolor="none",
                   markeredgecolor=plip_color, markersize=7, markeredgewidth=1.5,
                   label="PLIP hydrophobic"),
            Line2D([0], [0], marker="^", linestyle="", markerfacecolor="none",
                   markeredgecolor=plip_color, markersize=7, markeredgewidth=1.5,
                   label="PLIP halogen"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  framealpha=0.9, fontsize=9)

        if skipped:
            print(f"[plip] residues without struct mapping (skipped): {skipped}")

    fig.tight_layout()

    # Drug bar chart
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(drug_joint)), drug_joint)
    ax.set_xlabel("Drug atom index")
    ax.set_ylabel(f"Joint mass ({args.joint_mode})")
    ax.set_title(f"Drug per-atom joint mass - {pair_label}")
    if atom_labels is not None:
        _apply_sparse_ticks(ax, atom_labels, axis="x")
    fig.tight_layout()

    # Top-k printout
    k_prot = max(1, min(int(args.top_k), protein_joint.size))
    k_drug = max(1, min(int(args.top_k), drug_joint.size))
    prot_top = np.argsort(-protein_joint)[:k_prot]
    drug_top = np.argsort(-drug_joint)[:k_drug]

    print(f"\n==== Top protein indices by joint mass ({args.joint_mode}) ====")
    for rank, idx in enumerate(prot_top, 1):
        print(f"{rank:>2}. prot_idx={idx} (1-based {idx+1})  weight={float(protein_joint[idx]):.6g}")

    print(f"\n==== Top drug indices by joint mass ({args.joint_mode}) ====")
    for rank, jdx in enumerate(drug_top, 1):
        label = atom_labels[jdx] if atom_labels is not None else f"atom{jdx}"
        print(f"{rank:>2}. drug_idx={jdx} (1-based {jdx+1})  atom={label}  weight={float(drug_joint[jdx]):.6g}")

    plt.show()


if __name__ == "__main__":
    main()
