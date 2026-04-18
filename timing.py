import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.cfg import get_cfg_defaults
from dataset import MyDataset, collate_fn
from model import Model
import solver as solver_module


def sync_device(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize()
	elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
		torch.mps.synchronize()


def next_batch(it, loader):
	try:
		batch = next(it)
		return batch, it
	except StopIteration:
		it = iter(loader)
		batch = next(it)
		return batch, it


def to_device(batch, device: torch.device, non_blocking: bool):
	return {
		"label": batch["label"].to(device, non_blocking=non_blocking),
		"protein_mask": batch["protein_mask"].to(device, non_blocking=non_blocking),
		"drug_mask": batch["drug_mask"].to(device, non_blocking=non_blocking),
		"protein_emb": batch["protein_emb"].to(device, non_blocking=non_blocking),
		"drug_emb": batch["drug_emb"].to(device, non_blocking=non_blocking),
	}


def build_loader(cfg, batch_size: int, num_workers: int):
	ds = MyDataset(
		cfg.DATA.TRAIN_CSV_PATH,
		cfg.DATA.PROTEIN_DIR,
		cfg.DATA.DRUG_DIR,
		prot_cache_size=int(getattr(cfg.DATA, "PROT_CACHE_SIZE", 0)),
		drug_cache_size=int(getattr(cfg.DATA, "DRUG_CACHE_SIZE", 0)),
	)

	device_is_cuda = torch.cuda.is_available()
	kwargs = {
		"batch_size": batch_size,
		"shuffle": True,
		"num_workers": num_workers,
		"collate_fn": collate_fn,
		"drop_last": True,
		"pin_memory": bool(getattr(cfg.DATA, "PIN_MEMORY", True)) and device_is_cuda,
	}
	if num_workers > 0:
		kwargs["persistent_workers"] = bool(getattr(cfg.DATA, "PERSISTENT_WORKERS", True))
		kwargs["prefetch_factor"] = int(getattr(cfg.DATA, "PREFETCH_FACTOR", 2))

	return DataLoader(ds, **kwargs)


def get_loss_fn(cfg):
	if cfg.SOLVER.LOSS_FN == "dirichlet_loss":
		loss_fn = getattr(solver_module, "dirichlet_loss", None)
		if loss_fn is None:
			raise AttributeError("cfg.SOLVER.LOSS_FN is 'dirichlet_loss' but solver.dirichlet_loss was not found")
		return loss_fn
	return F.cross_entropy


def summarize_ms(values):
	ms = [v * 1000.0 for v in values]
	ms_sorted = sorted(ms)
	idx_95 = min(len(ms_sorted) - 1, int(0.95 * len(ms_sorted)))
	return {
		"mean": statistics.mean(ms),
		"median": statistics.median(ms),
		"p95": ms_sorted[idx_95],
		"min": min(ms),
		"max": max(ms),
	}


def main():
	parser = argparse.ArgumentParser(description="Benchmark training-step latency excluding DataLoader wait time.")
	parser.add_argument("--steps", type=int, default=50, help="Number of measured training steps.")
	parser.add_argument("--warmup", type=int, default=10, help="Warmup steps before measurement.")
	parser.add_argument("--batch_size", type=int, default=None, help="Override cfg.SOLVER.BATCH_SIZE.")
	parser.add_argument("--num_workers", type=int, default=None, help="Override cfg.DATA.NUM_WORKERS.")
	args = parser.parse_args()

	cfg = get_cfg_defaults()
	batch_size = args.batch_size if args.batch_size is not None else int(cfg.SOLVER.BATCH_SIZE)
	num_workers = args.num_workers if args.num_workers is not None else int(getattr(cfg.DATA, "NUM_WORKERS", 0))

	device = torch.device(
		"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
	)
	non_blocking = device.type == "cuda"

	model = Model(cfg).to(device)
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
	loss_fn = get_loss_fn(cfg)

	loader = build_loader(cfg, batch_size=batch_size, num_workers=num_workers)
	it = iter(loader)

	step_times = []
	load_wait_times = []

	total_iters = args.warmup + args.steps
	for i in range(total_iters):
		# Data loading wait time is measured separately and excluded from the training-step timer.
		t_load = time.perf_counter()
		batch, it = next_batch(it, loader)
		load_wait = time.perf_counter() - t_load

		sync_device(device)
		t_step = time.perf_counter()

		b = to_device(batch, device=device, non_blocking=non_blocking)
		optimizer.zero_grad(set_to_none=True)
		out = model(
			b["protein_emb"],
			b["drug_emb"],
			protein_mask=b["protein_mask"],
			drug_mask=b["drug_mask"],
			mode="train",
		)
		loss = loss_fn(out, b["label"])
		loss.backward()
		optimizer.step()

		sync_device(device)
		step_elapsed = time.perf_counter() - t_step

		if i >= args.warmup:
			step_times.append(step_elapsed)
			load_wait_times.append(load_wait)

	step_stats = summarize_ms(step_times)
	load_stats = summarize_ms(load_wait_times)

	print("=== Training-Step Timing (Data Loading Excluded) ===")
	print(f"Device: {device}")
	print(f"Measured steps: {args.steps} (warmup: {args.warmup})")
	print(
		"Step (no dataloader wait): "
		f"mean={step_stats['mean']:.3f} ms | median={step_stats['median']:.3f} ms | "
		f"p95={step_stats['p95']:.3f} ms | min={step_stats['min']:.3f} ms | max={step_stats['max']:.3f} ms"
	)
	print(
		"Dataloader wait (excluded): "
		f"mean={load_stats['mean']:.3f} ms | median={load_stats['median']:.3f} ms | "
		f"p95={load_stats['p95']:.3f} ms"
	)
	print(f"Effective no-load throughput: {(batch_size * 1000.0 / step_stats['mean']):.2f} samples/s")


if __name__ == "__main__":
	main()
