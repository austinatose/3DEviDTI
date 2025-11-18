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
        if k in ["protein_mask", "drug_mask", "label", "uniprot_id", "drugbank_id", "protein_lens", "drug_lens", "protein_mask", "drug_mask", "protein_emb", "drug_emb"]:
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


# check embeddings

import numpy as np
import hashlib

def tensor_hash(t: torch.Tensor) -> str:
    """Hash a tensor’s raw bytes for equality checks."""
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()

print("\n=== Embedding health check on 8 random samples ===")
indices = random.sample(range(len(ds)), min(8, len(ds)))

protein_means = []
protein_stds = []
drug_means = []
drug_stds = []
prot_hashes = []
drug_hashes = []
labels = []

for i in indices:
    item = ds[i]
    prot = item["protein_emb"]
    drug = item["drug_emb"]

    # flatten across L and D for simple stats
    protein_means.append(prot.mean().item())
    protein_stds.append(prot.std().item())
    drug_means.append(drug.mean().item())
    drug_stds.append(drug.std().item())

    prot_hashes.append(tensor_hash(prot))
    drug_hashes.append(tensor_hash(drug))
    labels.append(int(item["label"]))

    print(f"idx {i}:")
    print(f"  protein_id: {item['uniprot_id']}, drug_id: {item['drugbank_id']}, label: {item['label']}")
    print(f"  protein_emb: shape={tuple(prot.shape)}, mean={prot.mean():.4f}, std={prot.std():.4f}")
    print(f"  drug_emb   : shape={tuple(drug.shape)}, mean={drug.mean():.4f}, std={drug.std():.4f}")
    print(f"  protein hash: {prot_hashes[-1]}")
    print(f"  drug hash   : {drug_hashes[-1]}")
    print()

print("\n=== Summary stats over sampled embeddings ===")
print("Protein mean  (per-sample) -> mean {:.4f}, std {:.4f}".format(
    np.mean(protein_means), np.std(protein_means)))
print("Protein std   (per-sample) -> mean {:.4f}, std {:.4f}".format(
    np.mean(protein_stds), np.std(protein_stds)))
print("Drug mean     (per-sample) -> mean {:.4f}, std {:.4f}".format(
    np.mean(drug_means), np.std(drug_means)))
print("Drug std      (per-sample) -> mean {:.4f}, std {:.4f}".format(
    np.mean(drug_stds), np.std(drug_stds)))

print("\nUnique protein hashes:", len(set(prot_hashes)))
print("Unique drug hashes   :", len(set(drug_hashes)))
print("Labels in sample     :", labels)

# NaN/inf check on a batch from the DataLoader
print("\n=== Batch-level NaN/Inf check ===")
batch = next(iter(dl))
for name in ["protein_emb", "drug_emb"]:
    t = batch[name]
    print(f"{name}: shape={tuple(t.shape)}, "
          f"NaNs={torch.isnan(t).any().item()}, "
          f"Infs={torch.isinf(t).any().item()}, "
          f"mean={t.float().mean().item():.4f}, std={t.float().std().item():.4f}")