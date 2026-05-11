"""Benchmark ESM2 protein embeddings against the 3DICE model.

This script expects a CSV with columns:
  - unimol_encoding (drug embedding, 512-D per token)
  - target_embedding (protein embedding, e.g., 2560-D per token for ESM2-3B)
  - label (0/1)

Protein embeddings are downscaled to 512-D before being fed to the model.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import numpy as np
import re
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
	accuracy_score,
	matthews_corrcoef,
	roc_auc_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from config.cfg import get_cfg_defaults
from model import Model


class ESM2Dataset(Dataset):
	"""Dataset for CSVs where both embeddings are already stored as strings.

	Expected columns:
	  - unimol_encoding: drug embedding (either flat 1D of length d*L or 2D flattened)
	  - target_embedding: protein embedding (same format)
	  - label: {0,1}
	"""

	def __init__(
		self,
		csv_path: str | None = None,
		*,
		data: pd.DataFrame | None = None,
		emb_dim: int = 512,
		drug_emb_dim: int | None = None,
		protein_emb_dim: int | None = None,
	):
		self.drug_emb_dim = int(drug_emb_dim or emb_dim)
		self.protein_emb_dim = int(protein_emb_dim or emb_dim)

		if data is None:
			if csv_path is None:
				raise ValueError("Provide csv_path or data.")
			df = pd.read_csv(
				csv_path,
				usecols=["unimol_encoding", "target_embedding", "label"],
				dtype={"unimol_encoding": str, "target_embedding": str, "label": "int8"},
				engine="c",
				memory_map=True,
			)
		else:
			df = data

		self.drug_strs = df["unimol_encoding"].fillna("").tolist()
		self.prot_strs = df["target_embedding"].fillna("").tolist()
		self.labels = df["label"].astype("int8").tolist()
		if data is None:
			del df

	def __len__(self) -> int:
		return len(self.labels)

	@staticmethod
	def _parse_emb(s: str, emb_dim: int) -> torch.Tensor:
		if s is None:
			s = ""
		s = str(s).strip()
		if len(s) == 0:
			return torch.zeros((1, emb_dim), dtype=torch.float32)

		cleaned = s.replace("[", " ").replace("]", " ")
		cleaned = cleaned.replace("\n", " ").replace("\t", " ")
		nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cleaned)
		arr = np.asarray(nums, dtype=np.float32) if nums else np.asarray([], dtype=np.float32)

		if arr.size == 0:
			import ast
			try:
				obj = ast.literal_eval(s)
				arr = np.asarray(obj, dtype=np.float32).reshape(-1)
			except Exception:
				raise ValueError(f"Failed to parse embedding string (first 120 chars): {s[:120]!r}")

		if arr.size % emb_dim != 0:
			raise ValueError(
				f"Embedding length {arr.size} is not divisible by emb_dim={emb_dim}. "
				f"(first 120 chars): {s[:120]!r}"
			)
		arr = arr.reshape(-1, emb_dim)
		return torch.from_numpy(arr)

	def __getitem__(self, idx: int) -> dict:
		drug_emb = self._parse_emb(self.drug_strs[idx], self.drug_emb_dim)
		prot_emb = self._parse_emb(self.prot_strs[idx], self.protein_emb_dim)
		label = int(self.labels[idx])

		return {
			"protein_emb": prot_emb,
			"drug_emb": drug_emb,
			"label": label,
			"uniprot_id": None,
			"drug_id": None,
		}


def collate_esm2(batch: list[dict]) -> dict:
	prot_list = [b["protein_emb"] for b in batch]
	drug_list = [b["drug_emb"] for b in batch]
	labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

	prot_list = [torch.as_tensor(x).float() for x in prot_list]
	drug_list = [torch.as_tensor(x).float() for x in drug_list]

	prot_padded = pad_sequence(prot_list, batch_first=True)
	drug_padded = pad_sequence(drug_list, batch_first=True)

	B, Lp_max, _ = prot_padded.shape
	_, Ld_max, _ = drug_padded.shape
	prot_lens = torch.tensor([t.size(0) for t in prot_list])
	drug_lens = torch.tensor([t.size(0) for t in drug_list])

	prot_mask = torch.arange(Lp_max).unsqueeze(0).expand(B, Lp_max) >= prot_lens.unsqueeze(1)
	drug_mask = torch.arange(Ld_max).unsqueeze(0).expand(B, Ld_max) >= drug_lens.unsqueeze(1)

	return {
		"protein_emb": prot_padded,
		"drug_emb": drug_padded,
		"protein_mask": prot_mask,
		"drug_mask": drug_mask,
		"label": labels,
		"uniprot_id": [b.get("uniprot_id", None) for b in batch],
		"drug_id": [b.get("drug_id", None) for b in batch],
		"protein_lens": prot_lens,
		"drug_lens": drug_lens,
	}


class ProteinDownscaler(torch.nn.Module):
	def __init__(self, in_dim: int, out_dim: int, *, mode: str = "mean", seed: int = 0):
		super().__init__()
		self.in_dim = int(in_dim)
		self.out_dim = int(out_dim)
		self.mode = mode

		if self.in_dim == self.out_dim:
			self.proj = None
			return

		if self.mode == "linear":
			gen = torch.Generator()
			gen.manual_seed(seed)
			weight = torch.randn(self.out_dim, self.in_dim, generator=gen) * (1.0 / math.sqrt(self.in_dim))
			self.proj = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
			with torch.no_grad():
				self.proj.weight.copy_(weight)
			for p in self.proj.parameters():
				p.requires_grad_(False)
		else:
			self.proj = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.in_dim == self.out_dim:
			return x
		if self.mode == "mean" and self.in_dim % self.out_dim == 0:
			factor = self.in_dim // self.out_dim
			new_shape = (*x.shape[:-1], self.out_dim, factor)
			return x.view(new_shape).mean(dim=-1)
		if self.proj is None:
			raise ValueError(
				f"Cannot downscale from {self.in_dim} to {self.out_dim} with mode='{self.mode}'."
			)
		return self.proj(x)


def resolve_device(device_str: str) -> torch.device:
	if device_str == "auto":
		if torch.backends.mps.is_available():
			return torch.device("mps")
		if torch.cuda.is_available():
			return torch.device("cuda")
		return torch.device("cpu")
	return torch.device(device_str)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
		state = ckpt["model_state_dict"]
	else:
		state = ckpt
	model.load_state_dict(state)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
	if len(np.unique(y_true)) < 2:
		return float("nan")
	return roc_auc_score(y_true, y_score)


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
	if len(np.unique(y_true)) < 2:
		return float("nan")
	return average_precision_score(y_true, y_score)


def evaluate(
	model: torch.nn.Module,
	data_loader: DataLoader,
	device: torch.device,
	downscaler: ProteinDownscaler,
) -> dict[str, float]:
	return run_epoch(
		model,
		data_loader,
		device,
		downscaler,
		loss_fn=None,
		optimizer=None,
	)


def run_epoch(
	model: torch.nn.Module,
	data_loader: DataLoader,
	device: torch.device,
	downscaler: ProteinDownscaler,
	*,
	loss_fn: torch.nn.Module | None,
	optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
	is_train = optimizer is not None and loss_fn is not None
	model.train(is_train)
	all_labels: list[int] = []
	all_probs: list[float] = []
	all_preds: list[int] = []
	total_loss = 0.0
	steps = 0
	model.eval()
	context = torch.enable_grad() if is_train else torch.no_grad()
	with context:
		for batch in data_loader:
			labels = batch["label"].to(device)
			protein_mask = batch["protein_mask"].to(device)
			drug_mask = batch["drug_mask"].to(device)
			protein_emb = batch["protein_emb"].to(device)
			drug_emb = batch["drug_emb"].to(device)

			protein_emb = downscaler(protein_emb)

			logits = model(
				protein_emb,
				drug_emb,
				protein_mask=protein_mask,
				drug_mask=drug_mask,
				mode="train" if is_train else "test",
			)
			probs = F.softmax(logits, dim=1)
			preds = torch.argmax(probs, dim=1)

			if is_train:
				loss = loss_fn(logits, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += float(loss.item())
				steps += 1

			all_labels.extend(labels.detach().cpu().numpy().tolist())
			all_probs.extend(probs[:, 1].detach().cpu().numpy().tolist())
			all_preds.extend(preds.detach().cpu().numpy().tolist())

	y_true = np.asarray(all_labels)
	y_prob = np.asarray(all_probs)
	y_pred = np.asarray(all_preds)

	acc = accuracy_score(y_true, y_pred)
	mcc = matthews_corrcoef(y_true, y_pred) if len(y_true) else float("nan")
	auc = safe_auc(y_true, y_prob)
	auprc = safe_auprc(y_true, y_prob)
	prec = precision_score(y_true, y_pred, zero_division=0)
	rec = recall_score(y_true, y_pred, zero_division=0)
	f1 = f1_score(y_true, y_pred, zero_division=0)
	cm = confusion_matrix(y_true, y_pred) if len(y_true) else np.zeros((2, 2), dtype=int)
	avg_loss = (total_loss / steps) if steps else float("nan")

	print('Paste: %.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' % (acc, prec, rec, f1, mcc, auc, auprc))

	return {
		"loss": avg_loss,
		"accuracy": acc,
		"mcc": mcc,
		"auc": auc,
		"auprc": auprc,
		"precision": prec,
		"recall": rec,
		"f1": f1,
		"tn": float(cm[0, 0]),
		"fp": float(cm[0, 1]),
		"fn": float(cm[1, 0]),
		"tp": float(cm[1, 1]),
	}


def load_embeddings_table(
	emb_csv: str,
	*,
	drug_col: str,
	protein_col: str,
	drug_emb_col: str,
	protein_emb_col: str,
) -> pd.DataFrame:
	cols = [drug_col, protein_col, drug_emb_col, protein_emb_col]
	df = pd.read_csv(
		emb_csv,
		usecols=cols,
		dtype={drug_col: str, protein_col: str, drug_emb_col: str, protein_emb_col: str},
		engine="c",
		memory_map=True,
	)
	return df.rename(
		columns={
			drug_col: "drug_id",
			protein_col: "protein_id",
			drug_emb_col: "unimol_encoding",
			protein_emb_col: "target_embedding",
		}
	)


def build_split_df(
	split_csv: str,
	*,
	split_drug_col: str,
	split_protein_col: str,
	split_label_col: str,
	emb_df: pd.DataFrame,
) -> pd.DataFrame:
	split_df = pd.read_csv(
		split_csv,
		usecols=[split_drug_col, split_protein_col, split_label_col],
		dtype={split_drug_col: str, split_protein_col: str, split_label_col: "int8"},
		engine="c",
		memory_map=True,
	)
	split_df = split_df.rename(
		columns={
			split_drug_col: "drug_id",
			split_protein_col: "protein_id",
			split_label_col: "label",
		}
	)
	drug_map = emb_df[["drug_id", "unimol_encoding"]].dropna().drop_duplicates("drug_id")
	protein_map = emb_df[["protein_id", "target_embedding"]].dropna().drop_duplicates("protein_id")

	merged = split_df.merge(drug_map, on="drug_id", how="left")
	merged = merged.merge(protein_map, on="protein_id", how="left")

	missing_drug = int(merged["unimol_encoding"].isna().sum())
	missing_protein = int(merged["target_embedding"].isna().sum())
	if missing_drug > 0 or missing_protein > 0:
		print(
			f"[warn] {split_csv}: missing drug embeddings={missing_drug}, "
			f"missing protein embeddings={missing_protein}"
		)
	merged = merged.dropna(subset=["unimol_encoding", "target_embedding"]).reset_index(drop=True)
	if merged.empty:
		raise ValueError(f"No rows matched between {split_csv} and embeddings table.")
	return merged[["drug_id", "protein_id", "unimol_encoding", "target_embedding", "label"]]


def select_dataset_df(df: pd.DataFrame) -> pd.DataFrame:
	return df[["unimol_encoding", "target_embedding", "label"]]


def print_sanity_checks(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
	def summarize(name: str, df: pd.DataFrame) -> None:
		labels = df["label"].astype(int)
		pos = int(labels.sum())
		n = int(len(labels))
		neg = n - pos
		pos_rate = (pos / n) if n else float("nan")
		baseline = (max(pos, neg) / n) if n else float("nan")
		unique_drugs = df["drug_id"].nunique()
		unique_prots = df["protein_id"].nunique()
		print(
			f"[{name}] n={n} pos={pos} neg={neg} pos_rate={pos_rate:.4f} "
			f"majority_baseline={baseline:.4f} unique_drugs={unique_drugs} unique_proteins={unique_prots}"
		)

	summarize("train", train_df)
	summarize("val", val_df)
	summarize("test", test_df)

	train_pairs = set(zip(train_df["drug_id"], train_df["protein_id"]))
	val_pairs = set(zip(val_df["drug_id"], val_df["protein_id"]))
	test_pairs = set(zip(test_df["drug_id"], test_df["protein_id"]))

	print(f"[overlap] train∩val={len(train_pairs & val_pairs)}")
	print(f"[overlap] train∩test={len(train_pairs & test_pairs)}")
	print(f"[overlap] val∩test={len(val_pairs & test_pairs)}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark 3DICE with ESM2 protein embeddings.")
	parser.add_argument(
		"--emb-csv",
		default="DrugBank_uni_esm2_3B_safe.csv",
		help="ESM2 embeddings CSV with ids + unimol_encoding + target_embedding",
	)
	parser.add_argument("--csv", help="Alias for --emb-csv (deprecated)")
	parser.add_argument("--train-csv", required=True, help="Train split CSV with ids + label")
	parser.add_argument("--val-csv", required=True, help="Val split CSV with ids + label")
	parser.add_argument("--test-csv", required=True, help="Test split CSV with ids + label")
	parser.add_argument(
		"--ckpt",
		default="",
		help="Model checkpoint path",
	)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--weight-decay", type=float, default=0.0)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
	parser.add_argument("--drug-emb-dim", type=int, default=512)
	parser.add_argument("--protein-emb-dim", type=int, default=2560)
	parser.add_argument("--target-dim", type=int, default=512)
	parser.add_argument(
		"--downscale",
		choices=["mean", "linear"],
		default="mean",
		help="How to downscale protein embeddings to target-dim",
	)
	parser.add_argument("--emb-drug-col", default="0")
	parser.add_argument("--emb-protein-col", default="1")
	parser.add_argument("--emb-drug-emb-col", default="unimol_encoding")
	parser.add_argument("--emb-protein-emb-col", default="target_embedding")
	parser.add_argument("--split-drug-col", default="drug_id")
	parser.add_argument("--split-protein-col", default="uniprot_id")
	parser.add_argument("--split-label-col", default="interaction")
	parser.add_argument("--save-dir", default="saved")
	parser.add_argument("--best-dir", default="best_models")
	parser.add_argument("--log-dir", default="logs")
	parser.add_argument("--eval-best", action="store_true", help="Evaluate test using best val checkpoint")
	parser.add_argument("--num-workers", type=int, default=0)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.csv and not args.emb_csv:
		args.emb_csv = args.csv
	device = resolve_device(args.device)

	if args.drug_emb_dim != args.target_dim:
		raise ValueError(
			"drug_emb_dim must match target_dim because CrossAttention expects equal protein/drug dims."
		)
	if args.downscale == "mean" and args.protein_emb_dim % args.target_dim != 0:
		raise ValueError(
			"mean downscale requires protein_emb_dim to be divisible by target_dim. "
			"Use --downscale linear or adjust dims."
		)

	cfg = get_cfg_defaults()
	cfg.PROTEIN.EMBEDDING_DIM = int(args.target_dim)
	cfg.DRUG.EMBEDDING_DIM = int(args.target_dim)
	cfg.PROTEIN.INPUT_DIM = int(args.target_dim)
	cfg.MLP.INPUT_DIM = int(args.target_dim)
	cfg.SOLVER.DROPOUT = float(args.dropout)

	emb_df = load_embeddings_table(
		args.emb_csv,
		drug_col=args.emb_drug_col,
		protein_col=args.emb_protein_col,
		drug_emb_col=args.emb_drug_emb_col,
		protein_emb_col=args.emb_protein_emb_col,
	)

	train_df = build_split_df(
		args.train_csv,
		split_drug_col=args.split_drug_col,
		split_protein_col=args.split_protein_col,
		split_label_col=args.split_label_col,
		emb_df=emb_df,
	)
	val_df = build_split_df(
		args.val_csv,
		split_drug_col=args.split_drug_col,
		split_protein_col=args.split_protein_col,
		split_label_col=args.split_label_col,
		emb_df=emb_df,
	)
	test_df = build_split_df(
		args.test_csv,
		split_drug_col=args.split_drug_col,
		split_protein_col=args.split_protein_col,
		split_label_col=args.split_label_col,
		emb_df=emb_df,
	)

	print_sanity_checks(train_df, val_df, test_df)

	train_ds = ESM2Dataset(
		data=select_dataset_df(train_df),
		drug_emb_dim=args.drug_emb_dim,
		protein_emb_dim=args.protein_emb_dim,
	)
	val_ds = ESM2Dataset(
		data=select_dataset_df(val_df),
		drug_emb_dim=args.drug_emb_dim,
		protein_emb_dim=args.protein_emb_dim,
	)
	test_ds = ESM2Dataset(
		data=select_dataset_df(test_df),
		drug_emb_dim=args.drug_emb_dim,
		protein_emb_dim=args.protein_emb_dim,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		collate_fn=collate_esm2,
		drop_last=True,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_esm2,
		drop_last=False,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_esm2,
		drop_last=False,
	)

	model = Model(cfg).to(device)
	if args.ckpt:
		load_checkpoint(model, args.ckpt, device)

	downscaler = ProteinDownscaler(
		in_dim=args.protein_emb_dim,
		out_dim=args.target_dim,
		mode=args.downscale,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	loss_fn = torch.nn.CrossEntropyLoss()

	os.makedirs(args.save_dir, exist_ok=True)
	os.makedirs(args.best_dir, exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)

	start_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_path = os.path.join(args.log_dir, f"esm2_training_log_{start_stamp}.csv")
	with open(log_path, "w") as f:
		f.write(
			"epoch,train_loss,train_acc,val_acc,train_mcc,val_mcc,"
			"train_auc,val_auc,train_auprc,val_auprc,"
			"train_precision,val_precision,train_recall,val_recall,train_f1,val_f1\n"
		)

	best_val_acc = -1.0
	best_ckpt_path = ""

	for epoch in range(1, args.epochs + 1):
		train_metrics = run_epoch(
			model,
			train_loader,
			device,
			downscaler,
			loss_fn=loss_fn,
			optimizer=optimizer,
		)
		val_metrics = evaluate(model, val_loader, device, downscaler)

		print(
			"[Epoch %d] train loss: %.4f train acc: %.4f val acc: %.4f "
			"val auc: %.4f val auprc: %.4f val f1: %.4f"
			% (
				epoch,
				train_metrics["loss"],
				train_metrics["accuracy"],
				val_metrics["accuracy"],
				val_metrics["auc"],
				val_metrics["auprc"],
				val_metrics["f1"],
			)
		)

		ckpt_path = os.path.join(args.save_dir, f"esm2_epoch_{epoch}.pt")
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"cfg": cfg,
			},
			ckpt_path,
		)

		if val_metrics["accuracy"] > best_val_acc:
			best_val_acc = val_metrics["accuracy"]
			best_ckpt_path = os.path.join(args.best_dir, f"esm2_best_epoch_{epoch}.pt")
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"cfg": cfg,
				},
				best_ckpt_path,
			)

		with open(log_path, "a") as f:
			f.write(
				f"{epoch},{train_metrics['loss']:.6f},{train_metrics['accuracy']:.4f},"
				f"{val_metrics['accuracy']:.4f},{train_metrics['mcc']:.4f},"
				f"{val_metrics['mcc']:.4f},{train_metrics['auc']:.4f},{val_metrics['auc']:.4f},"
				f"{train_metrics['auprc']:.4f},{val_metrics['auprc']:.4f},"
				f"{train_metrics['precision']:.4f},{val_metrics['precision']:.4f},"
				f"{train_metrics['recall']:.4f},{val_metrics['recall']:.4f},"
				f"{train_metrics['f1']:.4f},{val_metrics['f1']:.4f}\n"
			)

	if args.eval_best and best_ckpt_path:
		load_checkpoint(model, best_ckpt_path, device)
		tag = "best"
	else:
		tag = "last"

	test_metrics = evaluate(model, test_loader, device, downscaler)
	print(f"Test metrics ({tag})")
	print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
	print(f"  MCC:       {test_metrics['mcc']:.4f}")
	print(f"  AUC:       {test_metrics['auc']:.4f}")
	print(f"  AUPRC:     {test_metrics['auprc']:.4f}")
	print(f"  Precision: {test_metrics['precision']:.4f}")
	print(f"  Recall:    {test_metrics['recall']:.4f}")
	print(f"  F1:        {test_metrics['f1']:.4f}")
	print(
		f"  TN/FP/FN/TP: {int(test_metrics['tn'])}/{int(test_metrics['fp'])}/"
		f"{int(test_metrics['fn'])}/{int(test_metrics['tp'])}"
	)


if __name__ == "__main__":
	main()
