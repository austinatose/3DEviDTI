

import time
import warnings

import numpy as np
import torch
from sklearn.metrics import (
	average_precision_score,
	f1_score,
	matthews_corrcoef,
	precision_score,
	recall_score,
	roc_auc_score,
)
from torch.utils.data import DataLoader
from dataset import MyDataset, collate_fn, KIBADataset
from model import Model
from config.cfg import get_cfg_defaults
from solver import Solver

# ---- config ----
# CKPT_PATH = "saved/cnn321_kiba_epoch_89.pt"  # replace XX with the epoch you want
# CKPT_PATH = "saved/cnn321_epoch_58.pt"
# model_4257565100199224673_epoch_36.pt

CKPT_PATH = "best_models/model_7607792504713019690_epoch_96.pt"

# Data loading performance knobs for on-demand .pt reads.
PROT_CACHE_SIZE = 4096
DRUG_CACHE_SIZE = 4096
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def sync_device(device):
	if device.type == "cuda":
		torch.cuda.synchronize()
	elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
		torch.mps.synchronize()


def time_forward_only(model, data_loader, device):
	forward_seconds = 0.0
	num_samples = 0
	num_batches = 0
	non_blocking = device.type == "cuda"

	model.eval()
	with torch.no_grad():
		for batch in data_loader:
			protein_mask = batch["protein_mask"].to(device, non_blocking=non_blocking)
			drug_mask = batch["drug_mask"].to(device, non_blocking=non_blocking)
			protein_emb = batch["protein_emb"].to(device, non_blocking=non_blocking)
			drug_emb = batch["drug_emb"].to(device, non_blocking=non_blocking)

			num_samples += batch["label"].shape[0]
			num_batches += 1

			sync_device(device)
			start = time.perf_counter()
			_ = model(
				protein_emb,
				drug_emb,
				protein_mask=protein_mask,
				drug_mask=drug_mask,
				mode="test",
			)
			sync_device(device)
			forward_seconds += time.perf_counter() - start

	forward_sps = num_samples / forward_seconds if forward_seconds > 0 else float("inf")
	forward_batch_ms = (forward_seconds / num_batches) * 1000 if num_batches > 0 else float("inf")
	forward_sample_ms = (forward_seconds / num_samples) * 1000 if num_samples > 0 else float("inf")

	return forward_seconds, forward_sps, forward_batch_ms, forward_sample_ms

# ---- load checkpoint ----
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
cfg = get_cfg_defaults()

model = Model(cfg=cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)

solver = Solver(model, cfg, device=DEVICE, optim=torch.optim.Adam, loss_fn=cfg.SOLVER.LOSS_FN, eval=None)

# test_ds = KIBADataset('lists/KIBA/KIBA_pairs_test.csv', cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
test_ds = MyDataset(
	'lists/db_new/db_test.csv',
	cfg.DATA.PROTEIN_DIR,
	cfg.DATA.DRUG_DIR,
	prot_cache_size=PROT_CACHE_SIZE,
	drug_cache_size=DRUG_CACHE_SIZE,
)

loader_kwargs = {
	"batch_size": cfg.SOLVER.BATCH_SIZE,
	"shuffle": False,
	"num_workers": NUM_WORKERS,
	"collate_fn": collate_fn,
	"drop_last": False,
	"pin_memory": DEVICE.type == "cuda",
}
if NUM_WORKERS > 0:
	loader_kwargs["persistent_workers"] = True
	loader_kwargs["prefetch_factor"] = 2

test_dl = DataLoader(test_ds, **loader_kwargs)

solver.model.eval()

with torch.no_grad():
	# Warm-up can reduce one-time startup overhead from skewing the timed run.
	_ = solver.predict_test(test_dl)

	start_time = time.perf_counter()
	_, _, pred_pairs, _, _, prob_list, _, _ = solver.predict_test(test_dl)
	elapsed_time = time.perf_counter() - start_time

# Separate pass for pure model-forward timing (excluding data loading + host->device transfer).
forward_seconds, forward_sps, forward_batch_ms, forward_sample_ms = time_forward_only(
	solver.model,
	test_dl,
	DEVICE,
)

val_results = np.squeeze(np.array(pred_pairs))
num_samples = len(val_results)
num_batches = len(test_dl)

val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / num_samples
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
	val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])

val_auc = roc_auc_score(val_results[:, 1], prob_list)
val_auprc = average_precision_score(val_results[:, 1], prob_list)
val_f1 = f1_score(val_results[:, 1], val_results[:, 0])
val_precision = precision_score(val_results[:, 1], val_results[:, 0])
val_recall = recall_score(val_results[:, 1], val_results[:, 0])

samples_per_sec = num_samples / elapsed_time if elapsed_time > 0 else float("inf")
avg_batch_ms = (elapsed_time / num_batches) * 1000 if num_batches > 0 else float("inf")
avg_sample_ms = (elapsed_time / num_samples) * 1000 if num_samples > 0 else float("inf")

print('Test accuracy: %.4f%% MCC: %.4f AUC: %.4f AUPRC: %.4f' % (val_acc, val_mcc, val_auc, val_auprc))
print('F1: %.4f Precision: %.4f Recall: %.4f' % (val_f1, val_precision, val_recall))
print('Inference timing: total %.4fs | %.2f samples/s | %.3f ms/batch | %.5f ms/sample' % (
	elapsed_time,
	samples_per_sec,
	avg_batch_ms,
	avg_sample_ms,
))
print('Forward-only timing: total %.4fs | %.2f samples/s | %.3f ms/batch | %.5f ms/sample' % (
	forward_seconds,
	forward_sps,
	forward_batch_ms,
	forward_sample_ms,
))
