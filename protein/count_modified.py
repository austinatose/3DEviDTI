

from pathlib import Path
import re
from tqdm import tqdm 
import json
import argparse
import os

 # CLI
parser = argparse.ArgumentParser(description="Scan structures for modified residues and write JSON index")
parser.add_argument("--root", type=str, default="structures", help="Root folder containing structure files")
parser.add_argument("--out", type=str, default="modified_files.json", help="Path to write JSON report")
args = parser.parse_args()

# Root folder containing structure files
STRUCT_ROOT = Path(args.root)

# List of 3-letter codes considered modified (same as SUBSTITUTIONS_3TO3 keys)
MODIFIED_CODES = {
    '2AS','3AH','5HP','ACL','AGM','AIB','ALM','ALO','ALY','ARM','ASA','ASB','ASK','ASL','ASQ','AYA','BCS','BHD','BMT','BNN','BUC','BUG','C5C','C6C','CAS','CCS','CEA','CGU','CHG','CLE','CME','CSD','CSO','CSP','CSS','CSW','CSX','CXM','CY1','CY3','CYG','CYM','CYQ','DAH','DAL','DAR','DAS','DCY','DGL','DGN','DHA','DHI','DIL','DIV','DLE','DLY','DNP','DPN','DPR','DSN','DSP','DTH','DTR','DTY','DVA','EFC','FLA','FME','GGL','GL3','GLZ','GMA','GSC','HAC','HAR','HIC','HIP','HMR','HPQ','HTR','HYP','IAS','IIL','IYR','KCX','LLP','LLY','LTR','LYM','LYZ','MAA','MEN','MHS','MIS','MLE','MPQ','MSA','MSE','MVA','NEM','NEP','NLE','NLN','NLP','NMC','OAS','OCS','OMT','PAQ','PCA','PEC','PHI','PHL','PR3','PRR','PTR','PYX','SAC','SAR','SCH','SCS','SCY','SEL','SEP','SET','SHC','SHR','SMC','SOC','STY','SVA','TIH','TPL','TPO','TPQ','TRG','TRO','TYB','TYI','TYQ','TYS','TYY'
}

# Compile regex pattern for quick search
pattern = re.compile(r'\b(' + '|'.join(sorted(MODIFIED_CODES, key=len, reverse=True)) + r')\b')

files = list(STRUCT_ROOT.rglob("*.cif")) + list(STRUCT_ROOT.rglob("*.pdb"))

total_files = len(files)
modified_files = 0
records = []
by_code = {code: 0 for code in MODIFIED_CODES}

for f in tqdm(files):
    try:
        with open(f, 'r', errors='ignore') as fh:
            text = fh.read()
        matches = list(pattern.finditer(text))
        if matches:
            modified_files += 1
            # Count per-code occurrences in this file
            from collections import Counter
            c = Counter(m.group(1) for m in matches)
            # Update global code counts
            for k, v in c.items():
                by_code[k] += v
            rel = str(f.relative_to(STRUCT_ROOT)) if f.is_relative_to(STRUCT_ROOT) else str(f)
            record = {
                "path": rel,
                "abs_path": str(f.resolve()),
                "ext": f.suffix.lower().lstrip('.'),
                "size_bytes": os.path.getsize(f),
                "modified_codes": sorted(c.keys()),
                "counts": dict(sorted(c.items())),
                "total_occurrences": int(sum(c.values())),
            }
            records.append(record)
    except Exception as e:
        print(f"[warn] Could not read {f}: {e}")

print(f"Total structure files: {total_files}")
print(f"Files containing modified residues: {modified_files}")
print(f"Fraction: {modified_files / total_files:.2%}")

# Compose JSON output
payload = {
    "root": str(STRUCT_ROOT.resolve()),
    "total_files": total_files,
    "files_with_modified": modified_files,
    "fraction": modified_files / total_files if total_files else 0.0,
    "code_totals": {k: v for k, v in sorted(by_code.items()) if v > 0},
    "records": sorted(records, key=lambda r: (r["path"]))
}

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False)

print(f"\nWrote JSON report → {out_path.resolve()}")