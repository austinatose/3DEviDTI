import pandas as pd
import os
import glob

INPUT_CSV = "lists/pairs_valid_cleaner.csv"
OUTPUT_CSV = "lists/only_clean_pairs.csv"
PROTEIN_EMB_DIR = "embeddings"

df = pd.read_csv(INPUT_CSV)

rows_with_pdb = []
missing = []

for idx, row in df.iterrows():
    protein_id = row["uniprot_id"]
    prot_dir = os.path.join(PROTEIN_EMB_DIR, protein_id)

    # skip if folder does not exist
    if not os.path.isdir(prot_dir):
        missing.append({"index": idx, "uniprot_id": protein_id, "reason": "no_dir"})
        continue

    # get all .pt files
    pt_files = glob.glob(os.path.join(prot_dir, "*.pt"))

    # filter ONLY pdb-based embeddings (those with "_" in filename)
    pdb_files = [f for f in pt_files if "_" in os.path.basename(f)]

    if len(pdb_files) == 0:
        # no pdb structure embeddings found
        missing.append({"index": idx, "uniprot_id": protein_id, "reason": "no_pdb_emb"})
        continue

    # keep the row; store which pdb embedding is used if needed
    row = row.copy()
    row["protein_emb_path"] = pdb_files[0]   # or choose by some priority rule
    rows_with_pdb.append(row)

df_pdb = pd.DataFrame(rows_with_pdb)
df_missing = pd.DataFrame(missing)

df_pdb.to_csv(OUTPUT_CSV, index=False)

print("Total:", len(df))
print("Kept (has pdb embedding):", len(df_pdb))
print("Missing:", len(df_missing))