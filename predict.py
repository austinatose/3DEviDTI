import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config.cfg import get_cfg_defaults
from model import Model

from tqdm import tqdm

def resolve_device(preferred: str) -> torch.device:
	if preferred and preferred != "auto":
		return torch.device(preferred)
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def load_protein_embedding(path: str) -> torch.Tensor:
	emb = torch.load(path, map_location="cpu", weights_only=True)
	if not isinstance(emb, torch.Tensor):
		emb = torch.as_tensor(emb)
	emb = emb.to(dtype=torch.float32, copy=False).contiguous()
	if emb.ndim != 2:
		raise ValueError(f"Protein embedding must be 2D (L, D), got shape {tuple(emb.shape)}")
	return emb


def load_drug_embedding(path: str) -> torch.Tensor:
	data = torch.load(path, map_location="cpu", weights_only=False)
	if isinstance(data, dict) and "atomic_reprs" in data:
		arr = np.asarray(data["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
		emb = torch.from_numpy(arr)
	else:
		emb = torch.as_tensor(data, dtype=torch.float32)
		if emb.ndim != 2:
			raise ValueError(f"Drug embedding must be 2D (L, D), got shape {tuple(emb.shape)}")
	return emb.contiguous()


def trim_sep(emb: torch.Tensor) -> torch.Tensor:
	return emb[:-1, :] if emb.size(0) > 1 else emb


def drug_id_from_path(path: str) -> str:
	name = os.path.basename(path)
	if name.endswith("_unimol.pt"):
		return name[: -len("_unimol.pt")]
	return os.path.splitext(name)[0]


def iter_drug_paths(drug_dir: str) -> list[str]:
	paths = glob.glob(os.path.join(drug_dir, "*.pt"))
	return sorted(paths)


def main() -> None:
	parser = argparse.ArgumentParser(description="Score one protein embedding against all drug embeddings.")
	parser.add_argument("--protein-emb", required=True, help="Path to a protein embedding .pt file")
	parser.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt file")
	parser.add_argument("--drug-dir", default="drug/embeddings_atomic", help="Folder with drug embedding .pt files")
	parser.add_argument("--out", default=None, help="CSV path to write positives (default: logs/predict_YYYYmmdd_HHMMSS.csv)")
	parser.add_argument("--batch-size", type=int, default=64, help="Number of drugs per batch")
	parser.add_argument("--threshold", type=float, default=0.5, help="Positive threshold on P(y=1)")
	parser.add_argument("--output-mode", choices=["ce", "dirichlet"], default="ce", help="How to interpret model outputs")
	parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
	args = parser.parse_args()

	device = resolve_device(args.device)

	ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
	cfg = get_cfg_defaults()
	model = Model(cfg=cfg)
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device)
	model.eval()

	protein_emb = trim_sep(load_protein_embedding(args.protein_emb))
	protein_len = protein_emb.size(0)

	drug_paths = iter_drug_paths(args.drug_dir)
	if not drug_paths:
		raise FileNotFoundError(f"No .pt files found in {args.drug_dir}")

	timestamp = time.strftime("%Y%m%d_%H%M%S")
	out_path = args.out or os.path.join("logs", f"predict_{drug_id_from_path(args.protein_emb)}_{timestamp}.csv")
	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	positives = 0
	total = 0
	positive_rows: list[tuple[str, float, str]] = []

	with torch.no_grad():
		for start in tqdm(range(0, len(drug_paths), args.batch_size)):
			batch_paths = drug_paths[start : start + args.batch_size]
			drug_embs = []
			drug_ids = []
			kept_paths = []

			for path in batch_paths:
				try:
					drug_emb = trim_sep(load_drug_embedding(path))
				except Exception as exc:
					print(f"[warn] skipping {path}: {exc}")
					continue
				drug_embs.append(drug_emb)
				drug_ids.append(drug_id_from_path(path))
				kept_paths.append(path)

			if not drug_embs:
				continue

			drug_padded = torch.nn.utils.rnn.pad_sequence(drug_embs, batch_first=True)
			drug_lens = torch.tensor([t.size(0) for t in drug_embs], dtype=torch.long)
			drug_mask = torch.arange(drug_padded.size(1)).unsqueeze(0) >= drug_lens.unsqueeze(1)

			protein_batch = protein_emb.unsqueeze(0).repeat(len(drug_embs), 1, 1)
			protein_mask = torch.zeros((len(drug_embs), protein_len), dtype=torch.bool)

			protein_batch = protein_batch.to(device)
			drug_padded = drug_padded.to(device)
			protein_mask = protein_mask.to(device)
			drug_mask = drug_mask.to(device)

			outputs = model(
				protein_batch,
				drug_padded,
				protein_mask=protein_mask,
				drug_mask=drug_mask,
				mode="test",
			)

			if args.output_mode == "ce":
				probs = F.softmax(outputs, dim=1)[:, 1]
			else:
				alphas = outputs
				sum_alpha = alphas.sum(dim=1)
				probs = alphas[:, 1] / sum_alpha

			probs = probs.detach().cpu().numpy().tolist()

			for drug_id, prob_pos, path in zip(drug_ids, probs, kept_paths):
				total += 1
				if prob_pos >= args.threshold:
					positives += 1
					positive_rows.append((drug_id, prob_pos, path))

	positive_rows.sort(key=lambda row: row[1], reverse=True)

	with open(out_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["drug_id", "prob_pos", "drug_path"])
		for drug_id, prob_pos, path in positive_rows:
			writer.writerow([drug_id, f"{prob_pos:.6f}", path])

	print(f"Processed {total} drugs. Positives: {positives}.")
	print(f"Positive log written to: {out_path}")


if __name__ == "__main__":
	main()
