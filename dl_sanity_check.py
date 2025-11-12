
# dl_sanity_check.py
#
# Quick, configurable sanity-check for a PyTorch Dataloader + Dataset.
#
# Usage examples:
#   python dl_sanity_check.py #       --csv lists/train.csv #       --module data.dataset #       --dataset-class DTIDataset #       --collate-fn collate_fn #       --batch-size 16 #       --dataset-kwargs '{"mode":"train","cache":false}'
#
#   python dl_sanity_check.py #       --csv data.csv #       --module src.data #       --dataset-class MyDS
#
# Notes:
# - Pass any constructor kwargs for your Dataset via --dataset-kwargs as a JSON string.
# - If your Dataset doesn't take a 'csv' path as the first arg, pass the right kwargs.
# - If your batches include masks or padding, you can give --pad-id to run extra checks.
# - Works whether your batch is a dict or a tuple/list.
import argparse
import importlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence

def set_seed(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def human_shape(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return f"Tensor{tuple(x.shape)} {x.dtype} device={x.device}"
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return f"ndarray{tuple(x.shape)} {x.dtype}"
    except Exception:
        pass
    if isinstance(x, (str, bytes, int, float, bool)):
        t = type(x).__name__
        return f"{t}"
    return type(x).__name__

def first_tensor(x):
    # Find the first tensor-like object in a nested batch to infer device/dtype.
    try:
        import torch
    except Exception:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, Mapping):
        for v in x.values():
            t = first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        for v in x:
            t = first_tensor(v)
            if t is not None:
                return t
    return None

def walk_items(batch):
    # Yield (path, value) pairs for nested structures (dict/list/tuple).
    def _walk(prefix, obj):
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                yield from _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for i, v in enumerate(obj):
                yield from _walk(f"{prefix}[{i}]" if prefix else f"[{i}]", v)
        else:
            yield prefix, obj
    yield from _walk("", batch)

def infer_label(batch):
    # Try to locate labels in a common field name.
    label_keys = ["label", "labels", "y", "interaction", "target"]
    if isinstance(batch, Mapping):
        for k in label_keys:
            if k in batch:
                return batch[k], k
    # If tuple/list, try last position
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)) and len(batch) > 0:
        return batch[-1], f"[{len(batch)-1}]"
    return None, None

def find_masks(batch):
    masks = {}
    if isinstance(batch, Mapping):
        for k, v in batch.items():
            if "mask" in k.lower():
                masks[k] = v
            elif isinstance(v, Mapping):
                for kk, vv in v.items():
                    if "mask" in kk.lower():
                        masks[f"{k}.{kk}"] = vv
    return masks

def check_no_nans_infs(x):
    import torch
    if isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
        if torch.isnan(x).any():
            return "NaNs present"
        if torch.isinf(x).any():
            return "Infs present"
    return None

def main():
    p = argparse.ArgumentParser(description="Sanity-check a PyTorch Dataset/DataLoader quickly.")
    p.add_argument("--csv", type=str, default=None, help="CSV or data path passed into Dataset (optional if covered in --dataset-kwargs).")
    p.add_argument("--module", type=str, required=True, help="Module path to import, e.g., 'data.dataset' or 'src.data'")
    p.add_argument("--dataset-class", type=str, required=True, help="Dataset class name in the module")
    p.add_argument("--collate-fn", type=str, default=None, help="Optional collate_fn name in the module")
    p.add_argument("--dataset-kwargs", type=str, default="{}", help="JSON dict of kwargs for Dataset(...)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=3, help="How many batches to iterate for checks")
    p.add_argument("--pad-id", type=int, default=None, help="Pad token/id, if you want padding checks where masks exist")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--drop-last", action="store_true")
    args = p.parse_args()

    # Imports
    try:
        torch = importlib.import_module("torch")
        from torch.utils.data import DataLoader
    except Exception as e:
        print("ERROR: PyTorch not importable. Please ensure torch is installed.\n", e)
        sys.exit(1)

    # Seed
    set_seed(args.seed)

    # Load module & symbols
    # try:
    #     mod = importlib.import_module(args.module)
    #     from 
    #     print(args.module, "imported.")
    # except Exception as e:
    #     print(f"ERROR: Could not import module '{args.module}'.\n{e}")
    #     sys.exit(1)

    # if not hasattr(mod, args.dataset_class):
    #     print(f"ERROR: Module '{args.module}' does not contain Dataset class '{args.dataset_class}'.")
    #     sys.exit(1)
    # DSCls = getattr(mod, args.dataset_class)

    # collate_fn = None
    # if args.collate_fn:
    #     if not hasattr(mod, args.collate_fn):
    #         print(f"ERROR: Module '{args.module}' does not contain collate_fn '{args.collate_fn}'.")
    #         sys.exit(1)
    #     collate_fn = getattr(mod, args.collate_fn)

    # Dataset kwargs
    try:
        ds_kwargs = json.loads(args.dataset_kwargs)
        if not isinstance(ds_kwargs, dict):
            raise ValueError("dataset-kwargs must decode to a JSON object")
    except Exception as e:
        print(f"ERROR: --dataset-kwargs must be JSON. Got: {args.dataset_kwargs}\n{e}")
        sys.exit(1)

    # Instantiate Dataset
    try:
        if args.csv is not None:
            ds = DSCls(args.csv, **ds_kwargs)
        else:
            ds = DSCls(**ds_kwargs)
    except TypeError as te:
        print("ERROR constructing Dataset. Check required args or pass via --dataset-kwargs.")
        print(te)
        sys.exit(1)

    n = len(ds) if hasattr(ds, "__len__") else None
    print(f"\n=== Dataset: {args.module}.{args.dataset_class}")
    print(f"Items: {n if n is not None else 'Unknown'}")
    print(f"Kwargs: {ds_kwargs}")
    if hasattr(ds, '__dict__'):
        print(f"Dataset attributes (truncated): {[k for k in list(ds.__dict__.keys())[:10]]}")

    # Basic indexing check
    try:
        _ = ds[0]
        print("\nIndexing ds[0] OK.")
    except Exception as e:
        print("\nERROR: Indexing ds[0] failed:", e)
        sys.exit(1)

    # Build DataLoader
    try:
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
            collate_fn=collate_fn,
        )
    except Exception as e:
        print("\nERROR: Building DataLoader failed:", e)
        sys.exit(1)

    # Iterate a few batches
    print(f"\n=== Iterating {args.num_batches} batch(es) ...")
    import time
    start = time.time()
    batches_seen = 0
    from math import ceil
    for i, batch in enumerate(dl):
        batches_seen += 1
        print(f"\n--- Batch {i+1} ---")
        # Overview of batch structure
        if isinstance(batch, Mapping):
            keys = list(batch.keys())
            print(f"Batch is dict with keys: {keys}")
            for k, v in batch.items():
                print(f"  {k}: {human_shape(v)}")
        elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            print(f"Batch is sequence (len={len(batch)})")
            for j, v in enumerate(batch):
                print(f"  [{j}]: {human_shape(v)}")
        else:
            print(f"Batch type: {type(batch).__name__}")

        # Infer device/dtype
        t0 = first_tensor(batch)
        if t0 is not None:
            print(f"First tensor device: {t0.device}, dtype: {t0.dtype}")

        # Labels sanity
        labels, label_path = infer_label(batch)
        if labels is not None:
            try:
                import torch
                if isinstance(labels, torch.Tensor):
                    unique = labels.unique(sorted=True).tolist()
                    print(f"Labels at {label_path}: unique values (up to 10): {unique[:10]}")
                    if labels.dtype in (torch.float16, torch.float32, torch.float64):
                        if torch.isnan(labels).any() or torch.isinf(labels).any():
                            print("!! Labels contain NaN/Inf")
                else:
                    try:
                        import numpy as np
                        unique = sorted(list(set(np.array(labels).tolist())))
                        print(f"Labels at {label_path}: uniques (sample): {unique[:10]}")
                    except Exception:
                        print(f"Labels at {label_path}: type {type(labels).__name__}")
            except Exception as e:
                print("Label check error:", e)

        # NaN/Inf checks on all tensor leaves
        nan_issues = []
        try:
            import torch
            for path, val in walk_items(batch):
                if isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
                    if torch.isnan(val).any():
                        nan_issues.append((path, "NaNs present"))
                    if torch.isinf(val).any():
                        nan_issues.append((path, "Infs present"))
            if nan_issues:
                print("!! Found NaN/Inf in:")
                for path, msg in nan_issues:
                    print(f"   {path}: {msg}")
        except Exception as e:
            print("Numeric check error:", e)

        # Optional padding/mask checks
        masks = find_masks(batch)
        if masks and args.pad_id is not None:
            print("Mask fields detected:", list(masks.keys()))
            # Heuristic: look for a sibling tensor with same shape to verify pad_id on masked positions
            def shape_of(x):
                try:
                    import torch
                    if isinstance(x, torch.Tensor):
                        return tuple(x.shape)
                except Exception:
                    return None
                return None
            # Gather all tensor leaves
            leaves = [(p, v) for p, v in walk_items(batch) if shape_of(v) is not None]
            for mkey, m in masks.items():
                mshape = shape_of(m)
                if not mshape:
                    continue
                # find candidate tensor with same shape
                candidate = None
                for pth, ten in leaves:
                    if pth == mkey:
                        continue
                    if shape_of(ten) == mshape:
                        candidate = (pth, ten)
                        break
                if candidate is None:
                    continue
                pth, ten = candidate
                import torch
                if m.dtype != torch.bool:
                    mask_bool = m != 0
                else:
                    mask_bool = m
                try:
                    neg_mask = ~mask_bool
                    if neg_mask.any():
                        pad_ok = (ten[neg_mask] == args.pad_id).all().item()
                    else:
                        pad_ok = True
                    print(f"Padding check vs '{pth}' using mask '{mkey}':", "OK" if pad_ok else "MISMATCH")
                except Exception as e:
                    print(f"Padding check failed for mask '{mkey}' and tensor '{pth}': {e}")

        if i + 1 >= args.num_batches:
            break

    dur = time.time() - start
    print(f"\nDone. Iterated {batches_seen} batch(es) in {dur:.2f}s.")
    if n is not None and args.batch_size:
        from math import ceil
        est_steps_per_epoch = ceil(n / args.batch_size)
        print(f"Est. steps/epoch @ batch_size={args.batch_size}: ~{est_steps_per_epoch}")
    print("\nIf you want deeper checks (leakage across splits, class balance), run your CSVs through a separate quickcheck.")
    print("Happy training!")

if __name__ == "__main__":
    main()
