import os
import re
import json
import glob
from pathlib import Path
from tqdm import tqdm

import torch
from esm import pretrained
from esm.inverse_folding.util import load_coords, get_encoder_output

# --- Canonicalize non‑standard residues for sequence generation -----------------
# Extend Biotite's 3→1 map so load_coords can create a 20‑AA sequence string.
# Geometry is unchanged; only the 1‑letter code used by the model is normalized.
try:
    from biotite.sequence.seqtypes import ProteinSequence
    _m = ProteinSequence._dict_3to1  # internal map used by convert_letter_3to1
    CANON_MAP = {
        # Canonical explicit
        "SER": "S", "DSN": "S",
        # Phospho‑residues
        "SEP": "S", "TPO": "T", "PTR": "Y",
        # Selenium / uncommon amino acids → closest canonical
        "MSE": "M", "SEC": "C", "PYL": "K",
        # Cys variants
        "CSO": "C", "CSD": "C", "CME": "C", "CYX": "C", "CSS": "C",
        # His protonation/tautomers
        "HIP": "H", "HIE": "H", "HID": "H",
        # Asp/Glu protonated
        "ASH": "D", "GLH": "E",
        # Oxidations / hydroxy
        "MHO": "M", "HYP": "P",
        # Common N/C‑terminal caps (ignored in sequence semantics)
        "ACE": "X", "NME": "X",
    }
    for k, v in CANON_MAP.items():
        _m.setdefault(k, v)
except Exception:
    pass
# -----------------------------------------------------------------------------

# -------------------- Config --------------------
STRUCT_ROOT = Path("structures")             # root folder containing your CIF/PDB files
EMB_ROOT = Path("embeddings")                # where to write outputs
SAVE_PER_FILE = True                          # save per-structure residue embeddings
AGGREGATE = False                              # write aggregated global embeddings
DEVICE = torch.device("cpu")                 # keep CPU for stability
POOL_STD = True                               # include std pooling (for 1024-D)

# ------------------------------------------------
EMB_ROOT.mkdir(parents=True, exist_ok=True)

# Load model + alphabet
model, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()  # ESM-IF1
model = model.to(DEVICE).eval()

# Utility: derive an id from a structure path
# Expected names like: structures/O00141/2R5T_A.cif -> id: O00141_2R5T_A
# ID_RE = re.compile(r"(?P<pdb>[0-9A-Za-z]{4})_(?P<chain>[A-Za-z0-9])\.(?:cif|pdb)$")

def derive_id(struct_path: Path) -> str:
    """Return the UniProt ID (parent folder name) for indexing."""
    parts = struct_path.parts
    # UniProt ID is typically the folder containing the structure file
    if len(parts) >= 2:
        return parts[-2]
    else:
        # fallback: filename without extension
        return struct_path.stem

# Pooling helpers
def pool_mean(emb: torch.Tensor) -> torch.Tensor:
    return emb.mean(dim=0)

def pool_mean_std(emb: torch.Tensor) -> torch.Tensor:
    mean = emb.mean(dim=0)
    std = emb.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=0)

# Enumerate structure files
files = sorted(
    list(STRUCT_ROOT.rglob("*.cif")) + list(STRUCT_ROOT.rglob("*.pdb"))
)

print(f"Found {len(files)} structure files under {STRUCT_ROOT}/")
if not files:
    print(f"No structures found under {STRUCT_ROOT}/. Nothing to do.")
    raise SystemExit(0)

ok_cnt = 0
skip_cnt = 0

index = []            # metadata rows
globals_512 = []      # list of 512-D tensors
globals_1024 = []     # list of 1024-D tensors (mean+std)

for i, fpath in enumerate(tqdm(files), 1):
    try:
        # Prefer letting util choose chain; if name has _X, try that as a fallback
        m_chain = re.search(r"_([A-Za-z0-9])\.(?:cif|pdb)$", fpath.name)
        tried = []
        last_err = None
        for attempt in ("named", "A", "X", "AAA"):
            try:
                if attempt == "named" and m_chain:
                    coords, seq = load_coords(str(fpath), chain=m_chain.group(1))
                elif attempt == "A":
                    coords, seq = load_coords(str(fpath), chain="A")
                elif attempt == "X":
                    coords, seq = load_coords(str(fpath), chain="X")
                elif attempt == "AAA":
                    coords, seq = load_coords(str(fpath), chain="AAA")
                else:
                    continue
                break  # success
            except Exception as e:
                last_err = e
                tried.append(attempt)
                coords = seq = None
        if coords is None or seq is None:
            raise RuntimeError(f"load_coords failed (tried {tried}): {last_err}")

        if "X" in seq:
            print(f"[warn] {fpath.name}: sequence contains 'X' (unknown residues); consider extending CANON_MAP if frequent.")
        # Extract residue embeddings [L, 512] using built-in helper
        rep = get_encoder_output(model, alphabet, coords)
        # Ensure CPU tensor
        rep = rep.detach().cpu()
        L = rep.shape[0]

        # Save per-structure embedding mirroring folder structure
        if SAVE_PER_FILE:
            rel = fpath.relative_to(STRUCT_ROOT)
            out_pt = EMB_ROOT.joinpath(rel).with_suffix(rel.suffix + ".pt")
            out_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rep, out_pt)
        else:
            out_pt = None

        # Aggregate
        if AGGREGATE:
            g512 = pool_mean(rep)  # [512]
            globals_512.append(g512.unsqueeze(0))
        if POOL_STD:
            g1024 = pool_mean_std(rep)  # [1024]
            globals_1024.append(g1024.unsqueeze(0))

        # Metadata row
        index.append({
            "id": derive_id(fpath),
            "path": str(fpath),
            "L": int(L),
            "out_pt": str(out_pt) if out_pt is not None else None,
        })
        ok_cnt += 1

    except Exception as e:
        print(f"[WARN] Skipping {fpath}: {e}")
        skip_cnt += 1
        continue

# Write aggregated tensors
if AGGREGATE and globals_512:
    G512 = torch.cat(globals_512, dim=0)  # [N, 512]
    torch.save(G512, EMB_ROOT / "all_global_512.pt")
if POOL_STD and globals_1024:
    G1024 = torch.cat(globals_1024, dim=0)  # [N, 1024]
    torch.save(G1024, EMB_ROOT / "all_global_1024.pt")
print(f"Wrote aggregated tensors to {EMB_ROOT}/")

# Write index metadata
with open(EMB_ROOT / "index.json", "w") as f:
    json.dump(index, f, indent=2)
print(f"Processed OK: {ok_cnt}  |  Skipped: {skip_cnt}  |  Total files: {len(files)}")
print(f"Wrote index with {len(index)} entries to {EMB_ROOT/'index.json'}")
