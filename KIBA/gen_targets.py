import re

input_path = "../data/Davis-KIBA/kiba.txt"
output_path = "../targets.csv"

# Regex for UniProt accessions (Swiss-Prot/TrEMBL primary):
UNIPROT_RE = re.compile(r"""
    (?:                                     # Two main accession formats
        [OPQ][0-9][A-Z0-9]{3}[0-9]          # O/P/Q prefix + 5 chars (e.g., O00141)
        |                                   # or
        [A-NR-Z][0-9]{5}                    # other letter + 5 digits (e.g., A0A123)
    )
""", re.X)

uids = set()
with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()                # split on any whitespace
        # Prefer column 2 if present; else regex fallback
        uid = parts[1] if len(parts) >= 2 else None
        if uid and UNIPROT_RE.fullmatch(uid):
            uids.add(uid)
        else:
            # fallback: search anywhere in the line for a UniProt-like token
            m = UNIPROT_RE.search(line)
            if m:
                uids.add(m.group(0))

with open(output_path, "w") as out_f:
    for uid in sorted(uids):
        out_f.write(f"{uid}\n")
