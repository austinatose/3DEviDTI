import numpy as np
from unimol_tools import UniMolRepr
import multiprocessing as mp
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
import os
import logging

import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# for name in ("unimol_tools", "Uni-Mol Tools", "unimol"):
#     lg = logging.getLogger(name)
#     lg.setLevel(logging.CRITICAL)
#     lg.propagate = False
#     if not lg.handlers:
#         lg.addHandler(logging.NullHandler())

RDLogger.DisableLog("rdApp.*")  # Disable all RDKit logs
import contextlib, io
f_out = io.StringIO()
f_err = io.StringIO()

CSV_PATH = "drug/drug_targets.csv"
OUTPUT_DIR = "drug/embeddings_atomic"

# iterate over every smiles in the csv and get unimol representation

def main():
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    # smiles = 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(O)=O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(N)=N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)CC1=CC=CC=C1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC1=CC=C(O)C=C1)C(=O)N[C@@H](CC(C)C)C(O)=O'
    
    df = pd.read_csv(CSV_PATH)
    smiles_list = df['SMILES'].tolist()
    id_list = df['drugbank_id'].tolist()
    # merge lists
    merged_list = list(zip(id_list, smiles_list))
    # unimol_reprs = clf.get_repr(CSV_PATH)
    for i, (drug_id, smi) in enumerate(tqdm(merged_list)):
        # skip existing files
        if os.path.exists(f"{OUTPUT_DIR}/{drug_id}_unimol.pt"):
            continue
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            atomic_reprs = clf.get_repr(smi, return_atomic_reprs=True)
        out_path = f"{OUTPUT_DIR}/{drug_id}_unimol.pt"
        torch.save(atomic_reprs, out_path)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

# save per-file because i will run out of ram