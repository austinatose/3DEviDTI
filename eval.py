

import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from dataset import MyDataset, collate_fn, KIBADataset
from model import Model
from config.cfg import get_cfg_defaults
from solver import Solver

# ---- config ----
CKPT_PATH = "saved/cnn321_kiba_epoch_89.pt"  # replace XX with the epoch you want
# model_4257565100199224673_epoch_36.pt
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cpu")

# ---- load checkpoint ----
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
cfg = get_cfg_defaults()

model = Model(cfg=cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)

solver = Solver(model, cfg, device=DEVICE, optim=torch.optim.Adam, loss_fn=cfg.SOLVER.LOSS_FN, eval=None)

test_ds = KIBADataset('lists/KIBA/KIBA_pairs_test.csv', cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
test_dl = DataLoader(test_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)

solver.evaluate(test_dl)