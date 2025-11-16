import torch
from torch.utils.data import DataLoader
from dataset import MyDataset 
from dataset import collate_fn
import os

# --- CONFIG --- #
csv_path = "lists/train.csv"               # your CSV file
protein_dir = "embeddings"         # folder with per-protein .pt files
drug_dir = "drug/embeddings_atomic"               # folder with per-drug .pt files
batch_size = 4
num_workers = 0

# --- LOAD DATASET --- #
ds = MyDataset(csv_path, protein_dir, drug_dir)
print(f"Dataset loaded. Total samples: {len(ds)}")

# Peek one item
print("\nSingle sample:")
s = ds[0]
for k, v in s.items():
    if torch.is_tensor(v):
        print(f"  {k}: {tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# --- TEST DATALOADER --- #
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last=True)

print("\nIterating a few batches...")
for i, batch in enumerate(dl):
    print(f"\nBatch {i+1}")
    for k, v in batch.items():
        if k in ["protein_mask", "drug_mask", "label", "uniprot_id", "drugbank_id", "protein_lens", "drug_lens"]:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {tuple(v.shape)}, dtype={v.dtype}")
    if i == 5:
        break

import random

for i in random.sample(range(len(ds)), 5):
    item = ds[i]
    print("idx:", i)
    print("  protein_id:", item["uniprot_id"])
    print("  drug_id:", item["drugbank_id"])
    print("  label:", item["label"])
    print("  protein_emb shape:", item["protein_emb"].shape)
    print("  drug_emb shape:", item["drug_emb"].shape)