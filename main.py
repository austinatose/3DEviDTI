import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from model import Model
from solver import Solver
# from config.cfg import get_cfg_defaults
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import glob
from functools import lru_cache

from tqdm import tqdm

def collate_fn(batch):
    # pull fields
    prot_list = [b["protein_emb"] for b in batch]   # each: (Lp, d)
    drug_list = [b["drug_emb"] for b in batch]      # each: (Ld, d)
    labels    = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    prot_ids  = [b["uniprot_id"] for b in batch]
    drugbank_ids  = [b["drugbank_id"] for b in batch]

    # ensure tensor dtype/shape
    prot_list = [torch.as_tensor(x).float() for x in prot_list]
    drug_list = [torch.as_tensor(x).float() for x in drug_list]

    # pad to max length within the batch (batch_first=True -> (B, Lmax, d))
    prot_padded = pad_sequence(prot_list, batch_first=True)   # (B, Lp_max, d)
    drug_padded = pad_sequence(drug_list, batch_first=True)   # (B, Ld_max, d)

    # build key_padding_masks: True = ignore (padding positions)
    B, Lp_max, _ = prot_padded.shape
    _, Ld_max, _ = drug_padded.shape
    prot_lens = torch.tensor([t.size(0) for t in prot_list])
    drug_lens = torch.tensor([t.size(0) for t in drug_list])

    prot_mask = torch.arange(Lp_max).unsqueeze(0).expand(B, Lp_max) >= prot_lens.unsqueeze(1)
    drug_mask = torch.arange(Ld_max).unsqueeze(0).expand(B, Ld_max) >= drug_lens.unsqueeze(1)

    return {
        "protein_emb": prot_padded,    # (B, Lp_max, d)
        "drug_emb": drug_padded,       # (B, Ld_max, d)
        "protein_mask": prot_mask,     # (B, Lp_max) bool, True=pad
        "drug_mask": drug_mask,        # (B, Ld_max) bool, True=pad
        "label": labels,               # (B,)
        "uniprot_id": prot_ids,
        "drugbank_id": drugbank_ids,
        "protein_lens": prot_lens,
        "drug_lens": drug_lens,
    }

def find_pt_files(emb_root, uniprot_id):
    pattern = os.path.join(emb_root, uniprot_id, "*.pt")
    return sorted(glob.glob(pattern))

class MyDataset(Dataset):
    def __init__(self, csv_path, protein_dir, drug_dir):
        self.df = pd.read_csv(csv_path)
        self.protein_dir = protein_dir
        self.drug_dir = drug_dir

        # Cache loaders (cache key will be the single argument: uniprot_id / drugbank_id)
        self._get_prot = lru_cache(maxsize=5000)(self._load_prot)
        self._get_drug = lru_cache(maxsize=5000)(self._load_drug)

    def _load_prot(self, uniprot_id):
        files = find_pt_files(self.protein_dir, str(uniprot_id))
        if not files:
            raise FileNotFoundError(f"No .pt files found for uniprot_id='{uniprot_id}' in {self.protein_dir}")
        # Use the last (alphabetically latest) match
        path = files[-1]
        emb = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(emb, dict) and "emb" in emb:
            emb = emb["emb"]
        return emb.float()

    def _load_drug(self, drugbank_id):
        path = os.path.join(self.drug_dir, f"{drugbank_id}_unimol.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Drug embedding not found: {path}")
        emb = torch.load(path, map_location="cpu", weights_only=False)
        emb = torch.FloatTensor(np.array(emb["atomic_reprs"]).reshape(-1, 512))
        return emb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uniprot_id = row["uniprot_id"]
        drugbank_id = row["drugbank_id"]
        label = int(row["interaction"])

        protein_emb = self._get_prot(uniprot_id)  # (L_p, d)
        drug_emb = self._get_drug(drugbank_id)        # (L_d, d)

        sample = {
            "protein_emb": protein_emb,
            "drug_emb": drug_emb,
            "label": label,
            "uniprot_id": uniprot_id,
            "drugbank_id": drugbank_id,
        }
        return sample

# def main():
#     cfg = get_cfg_defaults()
#     device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Model(cfg)

#     # use pre-split data first, then implement k-fold later

#     train_loader = DataLoader(, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)  # TODO: Initialize your DataLoader
#     val_loader = None    # TODO: Initialize your DataLoader
#     test_loader = None   # TODO: Initialize your DataLoader

#     solver = Solver(model, cfg, device, optim=torch.optim.Adam, loss_fn=torch.nn.CrossEntropyLoss(), eval=True) 
#     solver.train(train_loader, val_loader, test_loader)

