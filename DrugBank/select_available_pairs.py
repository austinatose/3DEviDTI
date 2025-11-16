# final step before finalising pair list
# we check all pairs in lists/pairs_filtered to check that both drugs and targets have available embeddings

import pandas as pd
import os
import torch

CSV_PATH = "lists/pairs_filtered_newer.csv"
DRUG_EMB_DIR = "drug/embeddings_atomic"
PROTEIN_EMB_DIR = "embeddings"

df = pd.read_csv(CSV_PATH)
filtered_pairs = []

for _, row in df.iterrows():
    drug_id = row['drugbank_id']
    protein_id = row['target_uniprot']

    drug_emb_path = os.path.join(DRUG_EMB_DIR, f"{drug_id}_unimol.pt")
    protein_emb_path = os.path.join(PROTEIN_EMB_DIR, f"{protein_id}")

    if os.path.exists(drug_emb_path) and os.path.exists(protein_emb_path) and os.listdir(protein_emb_path): # since failed protein embeddings result in empty folders
        filtered_pairs.append((drug_id, protein_id))

filtered_df = pd.DataFrame(filtered_pairs, columns=['drugbank_id', 'uniprot_id'])
filtered_df.to_csv("lists/pairs_valid_cleaner.csv", index=False)
print(f"Filtered pairs saved to lists/pairs_valid_cleaner.csv with {len(filtered_pairs)} pairs.")
