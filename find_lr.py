import os
import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from config.cfg import get_cfg_defaults
from dataset import stdDataset, collate_std

from model import stdModel
from torch_lr_finder import LRFinder


cfg = get_cfg_defaults()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = stdModel(cfg).to(device)

train_ds = stdDataset(
    cfg.DATA.TRAIN_CSV_PATH,
    cfg.DATA.PROTEIN_DIR,
    cfg.DATA.DRUG_DIR,
    prot_cache_size=int(getattr(cfg.DATA, "PROT_CACHE_SIZE", 0)),
    drug_cache_size=int(getattr(cfg.DATA, "DRUG_CACHE_SIZE", 0)),
)

num_workers = int(getattr(cfg.DATA, "NUM_WORKERS", 0))
loader_kwargs = {
    "batch_size": cfg.SOLVER.BATCH_SIZE,
    "shuffle": True,
    "num_workers": num_workers,
    "drop_last": True,
    "collate_fn": collate_std,
    "pin_memory": bool(getattr(cfg.DATA, "PIN_MEMORY", True)) and device == "cuda",
}
if num_workers > 0:
    loader_kwargs["persistent_workers"] = bool(getattr(cfg.DATA, "PERSISTENT_WORKERS", True))
    loader_kwargs["prefetch_factor"] = int(getattr(cfg.DATA, "PREFETCH_FACTOR", 2))

train_dl = DataLoader(train_ds, **loader_kwargs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=0) # weight decay?

lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
lrs = np.array(lr_finder.history["lr"])
losses = np.array(lr_finder.history["loss"])

min_grad_idx = None
best_lr = None
try:
    min_grad_idx = (np.gradient(np.array(losses))).argmin()
except ValueError:
    print("Failed to compute the gradients, there might not be enough points.")
if min_grad_idx is not None:
    best_lr = lrs[min_grad_idx]

print(f"Best lr:", best_lr)
lr_finder.reset() # to reset the model and optimizer to their initial state

