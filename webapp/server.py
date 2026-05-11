"""3DICE demo backend. Phase 1: single /api/predict endpoint."""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.cfg import get_cfg_defaults  # noqa: E402
from dataset import MyDataset  # noqa: E402
from model import Model  # noqa: E402
from webapp.structure_cache import ProteinStructure, load_structure, pdb_text_for_chain  # noqa: E402

CKPT_PATH = PROJECT_ROOT / "saved" / "model_6044491872843893979_epoch_50.pt"
SPLIT_CSVS = {
    "train": PROJECT_ROOT / "lists" / "db_train.csv",
    "val":   PROJECT_ROOT / "lists" / "db_val.csv",
    "test":  PROJECT_ROOT / "lists" / "db_test.csv",
}
PROTEIN_DIR = PROJECT_ROOT / "embeddings"
DRUG_DIR = PROJECT_ROOT / "drug" / "embeddings_atomic"
MAPPINGS_DIR = PROJECT_ROOT / "interpretation" / "mappings"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
STRUCT_CACHE_DIR = CACHE_DIR / "pdb"
DRUG_NAMES_PATH = CACHE_DIR / "drug_names.json"
PROTEIN_NAMES_PATH = CACHE_DIR / "protein_names.json"
FRONTEND_DIR = Path(__file__).resolve().parent / "static"
DEVICE = "cpu"

SPECIAL_TOKENS = {
    "SEP", "[SEP]", "<SEP>",
    "CLS", "[CLS]", "<CLS>",
    "BOS", "[BOS]", "<BOS>",
    "EOS", "[EOS]", "<EOS>",
}

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] loading checkpoint: {CKPT_PATH.name}")
    cfg = get_cfg_defaults()

    model = Model(cfg=cfg)
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    label_map: dict[tuple[str, str], dict] = {}
    for split, csv_path in SPLIT_CSVS.items():
        if not csv_path.exists():
            print(f"[startup] skipping missing split CSV: {csv_path}")
            continue
        print(f"[startup] loading split '{split}': {csv_path.relative_to(PROJECT_ROOT)}")
        df = pd.read_csv(csv_path, dtype={"drug_id": str, "uniprot_id": str})
        for r in df.itertuples():
            key = (str(r.drug_id), str(r.uniprot_id))
            # Test takes precedence over val/train if a pair somehow appears in multiple splits.
            if key not in label_map or split == "test":
                label_map[key] = {"label": int(r.interaction), "split": split}

    print(f"[startup] indexing embeddings under {PROTEIN_DIR.name}/ and {DRUG_DIR.name}/")
    # MyDataset only needs a CSV to seed its on-disk path indices; the fallback in _load_*
    # finds anything else by direct filename, so other splits' pairs still resolve.
    ds = MyDataset(str(SPLIT_CSVS["test"]), str(PROTEIN_DIR), str(DRUG_DIR))

    drug_emb_ids = {
        Path(f).stem.replace("_unimol", "")
        for f in os.listdir(DRUG_DIR)
        if f.endswith(".pt")
    }
    protein_emb_ids = {
        p for p in os.listdir(PROTEIN_DIR)
        if (PROTEIN_DIR / p).is_dir() and not p.startswith(".")
    }

    drug_names: dict[str, str] = (
        json.loads(DRUG_NAMES_PATH.read_text()) if DRUG_NAMES_PATH.exists() else {}
    )
    protein_names: dict[str, str] = (
        json.loads(PROTEIN_NAMES_PATH.read_text()) if PROTEIN_NAMES_PATH.exists() else {}
    )
    # Restrict the search index to entities we can actually run inference on.
    drug_index = [
        {"id": d, "name": drug_names.get(d, "")}
        for d in sorted(drug_emb_ids)
    ]
    protein_index = [
        {"id": p, "name": protein_names.get(p, "")}
        for p in sorted(protein_emb_ids)
    ]
    print(f"[startup] search index: {len(drug_index)} drugs, {len(protein_index)} proteins")

    STATE["cfg"] = cfg
    STATE["model"] = model
    STATE["ds"] = ds
    STATE["label_map"] = label_map
    STATE["structures"] = {}
    STATE["drug_index"] = drug_index
    STATE["protein_index"] = protein_index
    STATE["drug_names"] = drug_names
    STATE["protein_names"] = protein_names
    print(f"[startup] ready ({len(label_map)} pairs)")
    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan, title="3DICE demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_structure(uniprot_id: str) -> ProteinStructure:
    cache: dict[str, ProteinStructure] = STATE["structures"]
    if uniprot_id in cache:
        return cache[uniprot_id]
    map_dir = MAPPINGS_DIR / uniprot_id
    map_files = sorted(map_dir.glob("*.map.json"))
    if not map_files:
        raise HTTPException(404, f"no mapping JSON for uniprot_id={uniprot_id}")
    try:
        struct = load_structure(uniprot_id, map_files[0], STRUCT_CACHE_DIR)
    except Exception as e:
        raise HTTPException(404, f"structure load failed for {uniprot_id}: {e}")
    cache[uniprot_id] = struct
    return struct


@app.get("/api/health")
def health():
    lm = STATE.get("label_map", {})
    by_split: dict[str, int] = {}
    for v in lm.values():
        by_split[v["split"]] = by_split.get(v["split"], 0) + 1
    return {
        "status": "ok",
        "ckpt": CKPT_PATH.name,
        "n_pairs": len(lm),
        "by_split": by_split,
    }


@app.get("/api/search")
def search(
    type: str = Query(..., pattern="^(drug|protein)$"),
    q: str = Query("", max_length=64),
    limit: int = Query(20, ge=1, le=100),
):
    """Substring search on id and name (case-insensitive). Items where the query matches the
    id are listed first; then name matches."""
    index = STATE["drug_index"] if type == "drug" else STATE["protein_index"]
    q_lower = q.strip().lower()
    if not q_lower:
        return {"results": index[:limit]}
    id_hits: list[dict] = []
    name_hits: list[dict] = []
    for item in index:
        if q_lower in item["id"].lower():
            id_hits.append(item)
        elif item["name"] and q_lower in item["name"].lower():
            name_hits.append(item)
        if len(id_hits) + len(name_hits) >= limit * 3:
            break
    return {"results": (id_hits + name_hits)[:limit]}


def _load_drug_meta(drug_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (atomic_symbols, atomic_coords) from a UniMol .pt file."""
    raw = torch.load(drug_path, map_location="cpu", weights_only=False)
    symbols = np.array(raw["atomic_symbol"], dtype=object).reshape(-1)
    symbols = np.array([str(s).strip() for s in symbols], dtype=object)
    coords = np.array(raw["atomic_coords"], dtype=float)
    if coords.ndim >= 3:
        coords = np.squeeze(coords)
    if coords.ndim == 3 and coords.shape[-2:] == (3, 3):
        coords = coords[:, 1, :]
    return symbols, coords


@app.get("/api/predict")
def predict(
    drug_id: str = Query(...),
    uniprot_id: str = Query(...),
):
    ds: MyDataset = STATE["ds"]
    model: Model = STATE["model"]
    label_map: dict = STATE["label_map"]

    key = (str(drug_id), str(uniprot_id))
    pair_info = label_map.get(key)

    try:
        drug_emb = ds._get_drug(drug_id)
        prot_emb = ds._get_prot(uniprot_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    if ds.trim_terminal_token:
        if prot_emb.size(0) > 1:
            prot_emb = prot_emb[:-1, :]
        if drug_emb.size(0) > 1:
            drug_emb = drug_emb[:-1, :]

    Lp, Ld = prot_emb.size(0), drug_emb.size(0)
    prot_batch = prot_emb.unsqueeze(0).to(DEVICE)
    drug_batch = drug_emb.unsqueeze(0).to(DEVICE)
    prot_mask = torch.zeros(1, Lp, dtype=torch.bool, device=DEVICE)
    drug_mask = torch.zeros(1, Ld, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        logits, attn_p, attn_d = model(
            prot_batch, drug_batch,
            protein_mask=prot_mask, drug_mask=drug_mask,
            return_attention=True,
        )

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    attn_p_np = attn_p[0].cpu().numpy()  # (Lp, Ld)
    attn_d_np = attn_d[0].cpu().numpy()  # (Ld, Lp)

    drug_path = ds.drug_paths.get(str(drug_id))
    symbols, coords = _load_drug_meta(drug_path)
    n0 = min(len(symbols), coords.shape[0], Ld)
    symbols, coords = symbols[:n0], coords[:n0]

    # Drop SEP / dummy-zero-coord tokens consistently across symbols, coords, and the drug-axis of attention.
    syms_upper = np.array([s.upper() for s in symbols], dtype=object)
    is_special_sym = np.array(
        [u in SPECIAL_TOKENS or u == "" for u in syms_upper], dtype=bool
    )
    is_zero_coord = (
        np.all(np.isfinite(coords), axis=1)
        & (np.linalg.norm(coords, axis=1) < 1e-6)
    )
    special_mask = is_special_sym | is_zero_coord
    keep_idx = np.where(~special_mask)[0]
    if keep_idx.size == 0:
        raise HTTPException(500, "no atoms remained after filtering special tokens")

    attn_p_kept = attn_p_np[:, keep_idx]
    attn_d_kept = attn_d_np[keep_idx, :]
    symbols_kept = symbols[keep_idx].tolist()
    coords_kept = coords[keep_idx].tolist()

    # Structure is optional: an arbitrary pair may target a protein without a PDB mapping
    # or whose structure can't be fetched/aligned. In that case we still return the
    # prediction + attention; the frontend hides the 3D view.
    try:
        struct = _get_structure(uniprot_id)
    except Exception as e:
        print(f"[predict] no structure for {uniprot_id}: {e}")
        struct = None
    if struct is not None and Lp > len(struct.resnums):
        raise HTTPException(500, f"Lp={Lp} exceeds PDB residue count {len(struct.resnums)}")
    protein_resnums = struct.resnums[:Lp] if struct else None
    protein_icodes = struct.icodes[:Lp] if struct else None

    return {
        "drug_id": drug_id,
        "drug_name": STATE["drug_names"].get(drug_id, ""),
        "uniprot_id": uniprot_id,
        "protein_name": STATE["protein_names"].get(uniprot_id, ""),
        "pdb_id": struct.pdb_id if struct else None,
        "structure_source": struct.source if struct else None,
        "chain": struct.chain if struct else None,
        "prob_interaction": float(probs[1]),
        "pred_label": int(np.argmax(probs)),
        "true_label": pair_info["label"] if pair_info else None,
        "split": pair_info["split"] if pair_info else "unknown",
        "Lp": int(Lp),
        "Ld": int(len(keep_idx)),
        "attn_p": attn_p_kept.tolist(),
        "attn_d": attn_d_kept.tolist(),
        "atom_symbols": symbols_kept,
        "atom_coords": [[float(x), float(y), float(z)] for x, y, z in coords_kept],
        "protein_resnums": protein_resnums,
        "protein_icodes": protein_icodes,
    }


@app.get("/api/structure/{uniprot_id}", response_class=PlainTextResponse)
def structure(uniprot_id: str):
    struct = _get_structure(uniprot_id)
    return PlainTextResponse(
        pdb_text_for_chain(struct.pdb_text, struct.chain),
        media_type="chemical/x-pdb",
    )


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
