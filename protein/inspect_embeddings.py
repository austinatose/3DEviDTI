import torch
import numpy as np
from pathlib import Path

EMB_PATH = Path("embeddings")

all_files = list(EMB_PATH.rglob("*.pt"))
print(f"Found {len(all_files)} embedding files under {EMB_PATH}/")
uni_ids = sorted(set(p.parent.name for p in all_files))
print(f"Found embeddings for {len(uni_ids)} unique UniProt IDs.\n")

# --- Inspection utilities ---

def inspect_file(file_path):
    emb = torch.load(file_path, map_location="cpu")
    if isinstance(emb, dict):  # if you saved as dict from model outputs
        # try common keys like 'representations' or 'mean'
        if "representations" in emb:
            rep = emb["representations"]
            # sometimes it's dict of layers: pick last
            if isinstance(rep, dict):
                rep = rep[max(rep.keys())]
            emb = rep
        elif "mean" in emb:
            emb = emb["mean"]
        else:
            print(f"Unrecognized dict keys in {file_path.name}: {list(emb.keys())}")
            return

    arr = emb.numpy()
    print(f"→ {file_path.name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"   mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

# inspect first few
for f in all_files[:1]:
    inspect_file(f)

# optional: check embedding dimensions across all files
dims = []
for f in all_files:
    emb = torch.load(f, map_location="cpu")
    if isinstance(emb, dict):
        if "representations" in emb:
            emb = emb["representations"]
            if isinstance(emb, dict):
                emb = emb[max(emb.keys())]
        elif "mean" in emb:
            emb = emb["mean"]
    dims.append(tuple(emb.shape))

unique_dims = sorted(set(dims))
print("\nUnique embedding shapes found:", unique_dims)