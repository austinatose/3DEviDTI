# remove pairs from CSV that are too long
# FASTA referenced from uniprot_sequences

import argparse
import os
import time
import requests
from typing import Tuple
from tqdm import tqdm
import pandas as pd

MAX_PROT_LEN = 700
MAX_DRUG_LEN = 200

PRUNE_INPUT_PATH = "../MocFormer/DrugBank_uni_esm2_3B.csv"
PRUNE_OUTPUT_PATH = "../MocFormer/DrugBank_uni_esm2_3B_safe.csv"
DRUG_CHAIN_PATH = "lists/pairs_raw.csv"
PROT_CHAIN_PATH = "other/cache/uniprot_sequences.csv"

input_df = pd.read_csv(PRUNE_INPUT_PATH)
prot_df = pd.read_csv(PROT_CHAIN_PATH)
drug_df = pd.read_csv(DRUG_CHAIN_PATH)

def get_prot_chain(uniprot_id: str, prot_df: pd.DataFrame) -> str:
    row = prot_df[prot_df["uniprot_id"] == uniprot_id]
    if row.empty:
        return ' '*(MAX_PROT_LEN) # cook it
    return row.iloc[0]["sequence"]

def get_drug_chain(drug_id: str, drug_df: pd.DataFrame) -> str:
    row = drug_df[drug_df["drug_id"] == drug_id]
    if row.empty:
        return ' '*(MAX_DRUG_LEN) # cook it
    return row.iloc[0]["smiles"]

from pathlib import Path

def check_embed_existence(
    uniprot_id: str,
    drug_id: str,
    uniprot_embedding_path,
    drug_embedding_path,
) -> bool:
    """
    Protein: folder named <uniprot_id> exists AND contains at least one file
    Drug: file named <drug_id> exists
    """

    uniprot_embedding_path = Path(uniprot_embedding_path)
    drug_embedding_path = Path(drug_embedding_path)

    # ---- protein check: folder exists and not empty ----
    protein_dir = uniprot_embedding_path / uniprot_id
    protein_ok = (
        protein_dir.exists()
        and protein_dir.is_dir()
        and any(p.is_file() for p in protein_dir.iterdir())
    )

    # ---- drug check: file exists ----
    # check if file containing drug_id exists in drug_embedding_path
    # {drug_id}.pt or {drug_id}_unimol.pt
    drug_file_1 = drug_embedding_path / f"{drug_id}.pt"
    drug_file_2 = drug_embedding_path / f"{drug_id}_unimol.pt"
    drug_ok = drug_file_1.exists() or drug_file_2.exists()

    print(f"Protein {uniprot_id} OK: {protein_ok}, Drug {drug_id} OK: {drug_ok}")

    return protein_ok and drug_ok

print(input_df.columns.tolist())

for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
    # uniprot_id = row["0"]
    # drug_id = row["1"]

    # prot_chain = get_prot_chain(uniprot_id, prot_df)
    prot_chain = row['Target_sequence']
    # drug_chain = get_drug_chain(drug_id, drug_df)
    drug_chain = row['SMILES']

    if (len(prot_chain) > MAX_PROT_LEN or len(drug_chain) > MAX_DRUG_LEN) or prot_chain.isspace() or drug_chain.isspace() or not check_embed_existence(row['1'], row['0'], "embeddings", "drug/embeddings_atomic"):
        input_df.drop(index, inplace=True)

input_df.reset_index(drop=True, inplace=True)
input_df.to_csv(PRUNE_OUTPUT_PATH, index=False)

# %%
