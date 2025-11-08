import torch
from esm import pretrained
from esm.inverse_folding.util import load_coords, CoordBatchConverter, get_encoder_output

# --- Canonicalize non‑standard residues for sequence generation -----------------
# We extend Biotite's 3→1 map so load_coords can create a 20‑AA sequence string.
# Geometry is unchanged; only the 1‑letter code used by the model is normalized.
try:
    from biotite.sequence.seqtypes import ProteinSequence
    _m = ProteinSequence._dict_3to1  # internal map used by convert_letter_3to1
    # Core PTMs & variants → canonical parents
    CANON_MAP = {
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
        # Common N/C‑terminal caps (ignore as residue letters)
        "ACE": "X", "NME": "X",
    }
    for k, v in CANON_MAP.items():
        _m.setdefault(k, v)
except Exception:
    pass
# -----------------------------------------------------------------------------

# Select device (MPS if available, else CPU)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print(torch.ones(1, device=device))
# else:
#     print("MPS device not found.")
#     device = torch.device("cpu")

device = torch.device("cpu")

# Load model + alphabet
model, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50() ## ESM-IF1
model = model.to(device).eval()

# coords, seq = load_coords("structures/P39019/5VYC_T1.cif", chain="T1")  # temp
coords, seq = load_coords("structures/P47901/AF-P47901-F1-model.cif", chain="A") # AF uses A-chain by default

if "X" in seq:
    print(f"[warn] sequence contains 'X' (unknown residues). Consider extending CANON_MAP if frequent.")

rep = get_encoder_output(model, alphabet, coords)  # may allocate on MPS
print(rep.shape)

# the error is normal, i am not predicting contacts anyway