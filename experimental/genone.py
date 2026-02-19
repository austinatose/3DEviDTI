from matplotlib.pyplot import clf
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
elif torch.cuda.is_available():
    device = torch.device("cuda")
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
def main():
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles = 'CC1=C(SC=N1)C2=CC=C(C=C2)[C@H](C)NC(=O)[C@@H]3C[C@H](CN3C(=O)[C@H](C(C)(C)C)NC(=O)CCCCCC(=O)N4CCN(CC4)CC[C@H](CSC5=CC=CC=C5)NC6=C(C=C(C=C6)S(=O)(=O)NC(=O)C7=CC=C(C=C7)N8CCN(CC8)CC9=C(CCC(C9)(C)C)C1=CC=C(C=C1)Cl)S(=O)(=O)C(F)(F)F)O'
    drug_id = "Dt-2216"
    atomic_reprs = clf.get_repr(smiles, return_atomic_reprs=True)
    out_path = f"experimental/structures/{drug_id}.pt"
    torch.save(atomic_reprs, out_path)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

# save per-file because i will run out of ram