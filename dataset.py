import os
import glob
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # pull fields
    prot_list = [b["protein_emb"] for b in batch]   # each: (Lp, d)
    drug_list = [b["drug_emb"] for b in batch]      # each: (Ld, d)
    labels    = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    prot_ids  = [b["uniprot_id"] for b in batch]
    drug_ids  = [b["drug_id"] for b in batch]

    # ensure tensor dtype/shape
    prot_list = [torch.as_tensor(x).float() for x in prot_list]
    drug_list = [torch.as_tensor(x).float() for x in drug_list]

    # pad to max length within the batch (batch_first=True -> (B, Lmax, d))
    prot_padded = pad_sequence(prot_list, batch_first=True)   # (B, Lp_max, d)
    drug_padded = pad_sequence(drug_list, batch_first=True)   # (B, Ld_max, d)

    # build key_padding_masks: True = ignore (padding positions)
    B, Lp_max, _ = prot_padded.shape
    _, Ld_max, _ = drug_padded.shape
    prot_lens = torch.tensor([t.size(0) for t in prot_list])
    drug_lens = torch.tensor([t.size(0) for t in drug_list])

    prot_mask = torch.arange(Lp_max).unsqueeze(0).expand(B, Lp_max) >= prot_lens.unsqueeze(1)
    drug_mask = torch.arange(Ld_max).unsqueeze(0).expand(B, Ld_max) >= drug_lens.unsqueeze(1)

    return {
        "protein_emb": prot_padded,    # (B, Lp_max, d)
        "drug_emb": drug_padded,       # (B, Ld_max, d)
        "protein_mask": prot_mask,     # (B, Lp_max) bool, True=pad
        "drug_mask": drug_mask,        # (B, Ld_max) bool, True=pad
        "label": labels,               # (B,)
        "uniprot_id": prot_ids,
        "drug_id": drug_ids,
        "protein_lens": prot_lens,
        "drug_lens": drug_lens,
    }

def find_pt_files(emb_root, uniprot_id):
    pattern = os.path.join(emb_root, uniprot_id, "*.pt")
    return sorted(glob.glob(pattern))


class MyDataset(Dataset):
    def __init__(self, csv_path, protein_dir, drug_dir, *,
                 prot_cache_size: int = 0,
                 drug_cache_size: int = 0,
                 use_pandas: bool = True):
        """
        RAM-efficient dataset that lazily loads embeddings on demand.

        Args:
            csv_path: path to CSV with columns [uniprot_id, drug_id, interaction]
            protein_dir: root folder containing per-protein .pt embeddings under {uniprot_id}/*.pt
            drug_dir: folder containing drug embeddings saved as {drug_id}_unimol.pt
            prot_cache_size: LRU size for protein tensors (set small to keep RAM low; 0 disables caching)
            drug_cache_size: LRU size for drug tensors (set small to keep RAM low; 0 disables caching)
            use_pandas: if True, uses pandas to parse; else falls back to a lightweight CSV reader
        """
        self.protein_dir = protein_dir
        self.drug_dir = drug_dir

        # Parse CSV minimally and discard the DataFrame to free memory.
        if use_pandas:
            df = pd.read_csv(
                csv_path,
                usecols=["uniprot_id", "drug_id", "interaction"],
                dtype={"uniprot_id": str, "drug_id": str, "interaction": "int8"},
                engine="c",
                memory_map=True,
            )
            self.uniprot_ids = df["uniprot_id"].tolist()
            self.drug_ids = df["drug_id"].tolist()
            self.labels = df["interaction"].astype("int8").tolist()
            del df
        else:
            # Lightweight CSV parsing without pandas
            import csv
            self.uniprot_ids, self.drug_ids, self.labels = [], [], []
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.uniprot_ids.append(str(row["uniprot_id"]))
                    self.drug_ids.append(str(row["drug_id"]))
                    self.labels.append(int(row["interaction"]))

        # Configure tiny LRU caches (or disable if size == 0)
        if prot_cache_size > 0:
            self._get_prot = lru_cache(maxsize=prot_cache_size)(self._load_prot)
        else:
            self._get_prot = self._load_prot

        if drug_cache_size > 0:
            self._get_drug = lru_cache(maxsize=drug_cache_size)(self._load_drug)
        else:
            self._get_drug = self._load_drug

    def __len__(self):
        return len(self.uniprot_ids)

    def __getitem__(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        drug_id = self.drug_ids[idx]
        label = int(self.labels[idx])

        protein_emb = self._get_prot(uniprot_id)  # (L_p, d)
        protein_emb = protein_emb[:-1,:] if len(protein_emb) > 1 else protein_emb  # remove SEP
        drug_emb = self._get_drug(drug_id)    # (L_d, d)
        drug_emb = drug_emb[:-1,:] if len(drug_emb) > 1 else drug_emb  # remove SEP

        return {
            "protein_emb": protein_emb,
            "drug_emb": drug_emb,
            "label": label,
            "uniprot_id": uniprot_id,
            "drug_id": drug_id,
        }

    def _load_prot(self, uniprot_id: str):
        files = find_pt_files(self.protein_dir, str(uniprot_id))
        if not files:
            raise FileNotFoundError(
                f"No .pt files found for uniprot_id='{uniprot_id}' in {self.protein_dir}"
            )
        # Use the last (alphabetically latest) match
        path = files[-1]
        # If the file stores a plain tensor, weights_only=True avoids loading extraneous pickled objects
        emb = torch.load(path, map_location="cpu", weights_only=True)
        # Ensure float32 tensor and contiguous memory layout
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.to(dtype=torch.float32, copy=False).contiguous()
        return emb

    def _load_drug(self, drug_id: str):
        path = os.path.join(self.drug_dir, f"{drug_id}_unimol.pt")
        if not os.path.exists(path):
            path = os.path.join(self.drug_dir, f"{drug_id}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Drug embedding not found: {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
        arr = np.asarray(d["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
        # Drop the dict promptly to free RAM; keep only the tensor view
        del d
        emb = torch.from_numpy(arr)
        return emb

def collate_std(batch):
        # pull fields
    prot_list = [b[0]["protein_emb"] for b in batch]   # each: (Lp, d)
    drug_list = [b[0]["drug_emb"] for b in batch]      # each: (Ld, d)
    labels    = torch.tensor([b[1] for b in batch], dtype=torch.long)

    # ensure tensor dtype/shape
    prot_list = [torch.as_tensor(x).float() for x in prot_list]
    drug_list = [torch.as_tensor(x).float() for x in drug_list]

    # pad to max length within the batch (batch_first=True -> (B, Lmax, d))
    prot_padded = pad_sequence(prot_list, batch_first=True)   # (B, Lp_max, d)
    drug_padded = pad_sequence(drug_list, batch_first=True)   # (B, Ld_max, d)

    # build key_padding_masks: True = ignore (padding positions)
    B, Lp_max, _ = prot_padded.shape
    _, Ld_max, _ = drug_padded.shape
    prot_lens = torch.tensor([t.size(0) for t in prot_list])
    drug_lens = torch.tensor([t.size(0) for t in drug_list])

    prot_mask = torch.arange(Lp_max).unsqueeze(0).expand(B, Lp_max) >= prot_lens.unsqueeze(1)
    drug_mask = torch.arange(Ld_max).unsqueeze(0).expand(B, Ld_max) >= drug_lens.unsqueeze(1)

    return ({
        "protein_emb": prot_padded,
        "drug_emb": drug_padded,
        "protein_mask": prot_mask,
        "drug_mask": drug_mask,
    }, labels)

class stdDataset(Dataset):
    def __init__(self, csv_path, protein_dir, drug_dir, *,
                 prot_cache_size: int = 0,
                 drug_cache_size: int = 0,
                 use_pandas: bool = True):
        """
        RAM-efficient dataset that lazily loads embeddings on demand.

        Args:
            csv_path: path to CSV with columns [uniprot_id, drug_id, interaction]
            protein_dir: root folder containing per-protein .pt embeddings under {uniprot_id}/*.pt
            drug_dir: folder containing drug embeddings saved as {drug_id}_unimol.pt
            prot_cache_size: LRU size for protein tensors (set small to keep RAM low; 0 disables caching)
            drug_cache_size: LRU size for drug tensors (set small to keep RAM low; 0 disables caching)
            use_pandas: if True, uses pandas to parse; else falls back to a lightweight CSV reader
        """
        self.protein_dir = protein_dir
        self.drug_dir = drug_dir

        # Parse CSV minimally and discard the DataFrame to free memory.
        if use_pandas:
            df = pd.read_csv(
                csv_path,
                usecols=["uniprot_id", "drug_id", "interaction"],
                dtype={"uniprot_id": str, "drug_id": str, "interaction": "int8"},
                engine="c",
                memory_map=True,
            )
            self.uniprot_ids = df["uniprot_id"].tolist()
            self.drug_ids = df["drug_id"].tolist()
            self.labels = df["interaction"].astype("int8").tolist()
            del df
        else:
            # Lightweight CSV parsing without pandas
            import csv
            self.uniprot_ids, self.drug_ids, self.labels = [], [], []
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.uniprot_ids.append(str(row["uniprot_id"]))
                    self.drug_ids.append(str(row["drug_id"]))
                    self.labels.append(int(row["interaction"]))

        # Configure tiny LRU caches (or disable if size == 0)
        if prot_cache_size > 0:
            self._get_prot = lru_cache(maxsize=prot_cache_size)(self._load_prot)
        else:
            self._get_prot = self._load_prot

        if drug_cache_size > 0:
            self._get_drug = lru_cache(maxsize=drug_cache_size)(self._load_drug)
        else:
            self._get_drug = self._load_drug

    def __len__(self):
        return len(self.uniprot_ids)

    def __getitem__(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        drug_id = self.drug_ids[idx]
        label = int(self.labels[idx])

        protein_emb = self._get_prot(uniprot_id)  # (L_p, d)
        drug_emb = self._get_drug(drug_id)    # (L_d, d)

        return ({
            "protein_emb": protein_emb,
            "drug_emb": drug_emb,
            "drug_mask": None,
            "protein_mask": None,
        }, label)

    def _load_prot(self, uniprot_id: str):
        files = find_pt_files(self.protein_dir, str(uniprot_id))
        if not files:
            raise FileNotFoundError(
                f"No .pt files found for uniprot_id='{uniprot_id}' in {self.protein_dir}"
            )
        # Use the last (alphabetically latest) match
        path = files[-1]
        # If the file stores a plain tensor, weights_only=True avoids loading extraneous pickled objects
        emb = torch.load(path, map_location="cpu", weights_only=True)
        # Ensure float32 tensor and contiguous memory layout
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.to(dtype=torch.float32, copy=False).contiguous()
        return emb

    def _load_drug(self, drug_id: str):
        path = os.path.join(self.drug_dir, f"{drug_id}_unimol.pt")
        if not os.path.exists(path):
            path = os.path.join(self.drug_dir, f"{drug_id}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Drug embedding not found: {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
        arr = np.asarray(d["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
        # Drop the dict promptly to free RAM; keep only the tensor view
        del d
        emb = torch.from_numpy(arr)
        return emb

class KIBADataset(Dataset):
    def __init__(self, csv_path, protein_dir, drug_dir, *,
                 prot_cache_size: int = 0,
                 drug_cache_size: int = 0,
                 use_pandas: bool = True):

        self.protein_dir = protein_dir
        self.drug_dir = drug_dir

        # Parse CSV minimally and discard the DataFrame to free memory.
        if use_pandas:
            df = pd.read_csv(
                csv_path,
                usecols=["uniprot_id", "drug_id", "interaction"],
                dtype={"uniprot_id": str, "drug_id": str, "interaction": "int8"},
                engine="c",
                memory_map=True,
            )
            self.uniprot_ids = df["uniprot_id"].tolist()
            self.drug_ids = df["drug_id"].tolist()
            self.labels = df["interaction"].astype("int8").tolist()
            del df
        else:
            # Lightweight CSV parsing without pandas
            import csv
            self.uniprot_ids, self.drug_ids, self.labels = [], [], []
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.uniprot_ids.append(str(row["uniprot_id"]))
                    self.drug_ids.append(str(row["drug_id"]))
                    self.labels.append(int(row["interaction"]))

        # Configure tiny LRU caches (or disable if size == 0)
        if prot_cache_size > 0:
            self._get_prot = lru_cache(maxsize=prot_cache_size)(self._load_prot)
        else:
            self._get_prot = self._load_prot

        if drug_cache_size > 0:
            self._get_drug = lru_cache(maxsize=drug_cache_size)(self._load_drug)
        else:
            self._get_drug = self._load_drug

    def __len__(self):
        return len(self.uniprot_ids)

    def __getitem__(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        drug_id = self.drug_ids[idx]
        label = int(self.labels[idx])

        protein_emb = self._get_prot(uniprot_id)  # (L_p, d)
        drug_emb = self._get_drug(drug_id)    # (L_d, d)

        return {
            "protein_emb": protein_emb,
            "drug_emb": drug_emb,
            "label": label,
            "uniprot_id": uniprot_id,
            "drug_id": drug_id,
        }

    def _load_prot(self, uniprot_id: str):
        files = find_pt_files(self.protein_dir, str(uniprot_id))
        if not files:
            raise FileNotFoundError(
                f"No .pt files found for uniprot_id='{uniprot_id}' in {self.protein_dir}"
            )
        # Use the last (alphabetically latest) match
        path = files[-1]
        # If the file stores a plain tensor, weights_only=True avoids loading extraneous pickled objects
        emb = torch.load(path, map_location="cpu", weights_only=True)
        # Ensure float32 tensor and contiguous memory layout
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.to(dtype=torch.float32, copy=False).contiguous()
        return emb

    def _load_drug(self, drug_id: str):
        path = os.path.join(self.drug_dir, f"{drug_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Drug embedding not found: {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
        arr = np.asarray(d["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
        # Drop the dict promptly to free RAM; keep only the tensor view
        del d
        emb = torch.from_numpy(arr)
        return emb


class ESM2Dataset(Dataset):
    """Dataset for CSVs where both embeddings are already stored as strings.

    Expected columns (as currently used):
      - unimol_encoding: drug embedding (either flat 1D of length d*L or 2D flattened)
      - target_embedding: protein embedding (same format)
      - label: {0,1}

    The loader converts the string fields into float32 torch tensors.
    """

    def __init__(self, csv_path: str, *, emb_dim: int = 512):
        self.emb_dim = int(emb_dim)

        df = pd.read_csv(
            csv_path,
            usecols=["unimol_encoding", "target_embedding", "label"],
            dtype={"unimol_encoding": str, "target_embedding": str, "label": "int8"},
            engine="c",
            memory_map=True,
        )
        self.drug_strs = df["unimol_encoding"].fillna("").tolist()
        self.prot_strs = df["target_embedding"].fillna("").tolist()
        self.labels = df["label"].astype("int8").tolist()
        del df

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _parse_emb(s: str, emb_dim: int) -> torch.Tensor:
        """Parse a numeric embedding stored as a string into a (L, d) float32 tensor.

        Supports common formats:
        - space/comma separated numbers: "0.1 0.2 0.3 ..."
        - bracketed lists: "[0.1, 0.2, ...]"
        - nested lists: "[[...], [...]]" (will be flattened then reshaped)
        """
        if s is None:
            s = ""
        s = str(s).strip()
        if len(s) == 0:
            # empty -> return a single padding row to avoid zero-length tensors
            return torch.zeros((1, emb_dim), dtype=torch.float32)

        # Fast path: pull out numbers with numpy; works for many list/csv/space-separated strings.
        # Remove brackets to help fromstring when nested lists are present.
        cleaned = s.replace("[", " ").replace("]", " ")
        cleaned = cleaned.replace("\n", " ").replace("\t", " ")
        arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)
        if arr.size == 0:
            # Try comma-separated
            arr = np.fromstring(cleaned.replace(",", " "), sep=" ", dtype=np.float32)

        if arr.size == 0:
            # Fallback: python literal parsing (slower but robust)
            import ast
            try:
                obj = ast.literal_eval(s)
                arr = np.asarray(obj, dtype=np.float32).reshape(-1)
            except Exception:
                raise ValueError(f"Failed to parse embedding string (first 120 chars): {s[:120]!r}")

        # Ensure we can reshape into (-1, emb_dim)
        if arr.size % emb_dim != 0:
            raise ValueError(
                f"Embedding length {arr.size} is not divisible by emb_dim={emb_dim}. "
                f"(first 120 chars): {s[:120]!r}"
            )
        arr = arr.reshape(-1, emb_dim)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        drug_emb = self._parse_emb(self.drug_strs[idx], self.emb_dim)
        prot_emb = self._parse_emb(self.prot_strs[idx], self.emb_dim)
        label = int(self.labels[idx])

        return {
            "protein_emb": prot_emb,
            "drug_emb": drug_emb,
            "label": label,
            "uniprot_id": None,
            "drug_id": None,
        }


def collate_esm2(batch):
    """Collate for ESM2Dataset outputs.

    Pads variable-length protein/drug sequences to the max length in the batch and
    returns key_padding_masks (True = padding).

    Returns the same keys as `collate_fn` so you can reuse the same model forward.
    """
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
