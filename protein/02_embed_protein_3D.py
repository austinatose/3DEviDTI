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
    from biotite.sequence import ProteinSequence
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

    # --- Add user's 3→3 substitutions and derive 3→1 tokens for ESM -------------
    SUBSTITUTIONS_3TO3 = {
        '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
        'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
        'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
        'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
        'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
        'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
        'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
        'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
        'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
        'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
        'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
        'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
        'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
        'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
    }

    # Canonical 3-letter → 1-letter mapping for targets
    _STD_3TO1 = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
        'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
        'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }

    # Derive 3→1 entries from the 3→3 table
    _added_subs = 0
    for k3, v3 in SUBSTITUTIONS_3TO3.items():
        v3 = v3.upper()
        v1 = _STD_3TO1.get(v3)
        if v1 and k3 not in _m:
            _m[k3] = v1
            _added_subs += 1
    # Optional debug: uncomment to inspect how many entries were added
    # print(f"[canon] added {_added_subs} entries derived from SUBSTITUTIONS_3TO3")
    # ---------------------------------------------------------------------------
except Exception:
    pass
# -----------------------------------------------------------------------------

# -------------------- Config --------------------
STRUCT_ROOT = Path("structures")             # root folder containing your CIF/PDB files
EMB_ROOT = Path("embeddings")                # where to write outputs
SAVE_PER_FILE = True                          # save per-structure residue embeddings
AGGREGATE = False                              # write aggregated global embeddings
DEVICE = torch.device("cpu")                 # keep CPU for stability
POOL_STD = False                               # include std pooling (for 1024-D)

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
        # Determine output path early and skip if already embedded
        rel = fpath.relative_to(STRUCT_ROOT)
        out_pt = EMB_ROOT.joinpath(rel).with_suffix(rel.suffix + ".pt")
        if out_pt.exists():
            print(f"[skip] {fpath.name}: embedding already exists → {out_pt}")
            skip_cnt += 1
            continue

        # Determine chain based on filename and filetype
        if fpath.suffix == ".cif" and "AF-" in fpath.name:
            chain = "A"
        else:
            m_chain = re.search(r"_(?P<chain>[A-Za-z0-9]+)\.(?:cif|pdb)$", fpath.name)
            if m_chain:
                chain = m_chain.group("chain")
            else:
                chain = "A"
        coords, seq = load_coords(str(fpath), chain=chain)

        if "X" in seq:
            print(f"[warn] {fpath.name}: sequence contains 'X' (unknown residues); consider extending CANON_MAP if frequent.")
        # Extract residue embeddings [L, 512] using built-in helper
        rep = get_encoder_output(model, alphabet, coords)
        # Ensure CPU tensor
        rep = rep.detach().cpu()
        L = rep.shape[0]

        # Save per-structure embedding mirroring folder structure
        if SAVE_PER_FILE:
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
