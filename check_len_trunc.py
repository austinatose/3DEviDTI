import os
import glob
import numpy as np
import pandas as pd
import torch

from config.cfg import get_cfg_defaults
from dataset import MyDataset
from model import Model


def main():
    cfg = get_cfg_defaults()

    split_paths = [cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.VAL_CSV_PATH, cfg.DATA.TEST_CSV_PATH]
    all_df = pd.concat(
        [pd.read_csv(p, usecols=["drug_id", "uniprot_id", "interaction"]) for p in split_paths],
        ignore_index=True,
    )
    prot_ids = sorted(all_df["uniprot_id"].astype(str).unique().tolist())
    drug_ids = sorted(all_df["drug_id"].astype(str).unique().tolist())

    prot_len = {}
    for pid in prot_ids:
        files = sorted(glob.glob(os.path.join(cfg.DATA.PROTEIN_DIR, pid, "*.pt")))
        t = torch.load(files[-1], map_location="cpu", weights_only=True)
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        prot_len[pid] = int(t.shape[0])

    drug_len = {}
    for did in drug_ids:
        p1 = os.path.join(cfg.DATA.DRUG_DIR, f"{did}_unimol.pt")
        p2 = os.path.join(cfg.DATA.DRUG_DIR, f"{did}.pt")
        p = p1 if os.path.exists(p1) else p2
        d = torch.load(p, map_location="cpu", weights_only=False)
        arr = np.asarray(d["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
        drug_len[did] = int(arr.shape[0])

    max_prot_id = max(prot_len, key=prot_len.get)
    max_drug_id = max(drug_len, key=drug_len.get)
    print("Max file lengths:")
    print("  protein:", max_prot_id, prot_len[max_prot_id])
    print("  drug   :", max_drug_id, drug_len[max_drug_id])
    print("  proteins >700:", sum(1 for v in prot_len.values() if v > 700))
    print("  drugs >200   :", sum(1 for v in drug_len.values() if v > 200))

    ds = MyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    train_df = pd.read_csv(cfg.DATA.TRAIN_CSV_PATH, usecols=["drug_id", "uniprot_id"])

    prot_rows = train_df.index[train_df["uniprot_id"].astype(str) == max_prot_id].tolist()
    drug_rows = train_df.index[train_df["drug_id"].astype(str) == max_drug_id].tolist()

    if prot_rows:
        i = int(prot_rows[0])
        sample = ds[i]
        p_len_runtime = int(sample["protein_emb"].shape[0])
        raw = prot_len[max_prot_id]
        print("\nLongest-protein sample in train:")
        print("  raw file len      =", raw)
        print("  dataset returned  =", p_len_runtime)
        print("  expected returned =", raw - 1 if raw > 1 else raw)

    if drug_rows:
        i = int(drug_rows[0])
        sample = ds[i]
        d_len_runtime = int(sample["drug_emb"].shape[0])
        raw = drug_len[max_drug_id]
        print("\nLongest-drug sample in train:")
        print("  raw file len      =", raw)
        print("  dataset returned  =", d_len_runtime)
        print("  expected returned =", raw - 1 if raw > 1 else raw)

    # run forward pass on hardest-length train sample
    train_df["prot_raw_len"] = train_df["uniprot_id"].astype(str).map(prot_len)
    train_df["drug_raw_len"] = train_df["drug_id"].astype(str).map(drug_len)
    row_idx = int((train_df["prot_raw_len"] + train_df["drug_raw_len"]).idxmax())
    s = ds[row_idx]

    protein = s["protein_emb"].unsqueeze(0)
    drug = s["drug_emb"].unsqueeze(0)
    b, lp, _ = protein.shape
    _, ld, _ = drug.shape
    prot_mask = torch.zeros((b, lp), dtype=torch.bool)
    drug_mask = torch.zeros((b, ld), dtype=torch.bool)

    model = Model(cfg)
    model.eval()
    with torch.no_grad():
        out = model(protein, drug, protein_mask=prot_mask, drug_mask=drug_mask, mode="test")
    print("\nForward pass on longest train sample: OK, output shape =", tuple(out.shape))


if __name__ == "__main__":
    main()
