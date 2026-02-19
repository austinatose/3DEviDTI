import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch

from config.cfg import get_cfg_defaults
from model import Model


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
        default="geom",
        choices=["geom", "arith", "harm", "prod", "min"],
    )
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    device = torch.device(args.device)

    protein_emb = _load_protein_embedding(args.protein_emb)
    drug_emb, atom_labels = _load_drug_embedding(args.drug_emb)

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
