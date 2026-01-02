

"""translate_index.py

Utility to translate between:
  - STRUCT indices: indices into `seq_struct` returned by ESM-IF1 `load_coords()` (0-based)
  - FASTA indices: indices into the UniProt/FASTA sequence (0-based)

This script loads the `.map.json` files produced by `interpretation/biotite_mapping.py`.

The JSON schema (written by biotite_mapping.py) includes:
  - struct_to_fasta: list length L_struct with entries either an int (0-based fasta index) or null
  - fasta_to_struct: list length L_fasta with entries either an int (0-based struct index) or null

Examples
--------
Translate a struct index -> fasta index:
  python interpretation/translate_index.py \
    --map-json interpretation/mappings/O00141/2R5T_A.map.json \
    --struct-idx 193

Translate using the structure path (auto-resolve the mapping JSON by mirroring paths):
  python interpretation/translate_index.py \
    --struct-path structures/O00141/2R5T_A.cif \
    --struct-root structures \
    --map-root interpretation/mappings \
    --struct-idx 193

Reverse (fasta -> struct):
  python interpretation/translate_index.py \
    --map-json interpretation/mappings/O00141/2R5T_A.map.json \
    --fasta-idx 276

If you use 1-based indices (common in papers), pass --one-based.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Any, Dict


@dataclass
class IndexMapping:
    """Holds the mapping arrays produced by biotite_mapping.py."""

    map_json: Path
    struct_to_fasta: List[Optional[int]]
    fasta_to_struct: List[Optional[int]]

    id: Optional[str] = None
    chain: Optional[str] = None
    structure_path: Optional[str] = None
    fasta_path: Optional[str] = None

    def struct_idx_to_fasta_idx(self, struct_idx: int) -> Optional[int]:
        if struct_idx < 0 or struct_idx >= len(self.struct_to_fasta):
            raise IndexError(f"struct_idx={struct_idx} out of range [0, {len(self.struct_to_fasta)-1}]")
        return self.struct_to_fasta[struct_idx]

    def fasta_idx_to_struct_idx(self, fasta_idx: int) -> Optional[int]:
        if fasta_idx < 0 or fasta_idx >= len(self.fasta_to_struct):
            raise IndexError(f"fasta_idx={fasta_idx} out of range [0, {len(self.fasta_to_struct)-1}]")
        return self.fasta_to_struct[fasta_idx]


def load_map_json(map_json: Path) -> IndexMapping:
    """Load a `.map.json` produced by biotite_mapping.py."""
    with map_json.open("r", encoding="utf-8") as f:
        payload: Dict[str, Any] = json.load(f)

    stf = payload.get("struct_to_fasta")
    fts = payload.get("fasta_to_struct")
    if not isinstance(stf, list) or not isinstance(fts, list):
        raise ValueError(f"Invalid mapping JSON (missing arrays): {map_json}")

    # Normalize nulls to None and ensure ints where present
    def _norm(arr: List[Any]) -> List[Optional[int]]:
        out: List[Optional[int]] = []
        for x in arr:
            if x is None:
                out.append(None)
            else:
                try:
                    out.append(int(x))
                except Exception:
                    out.append(None)
        return out

    return IndexMapping(
        map_json=map_json,
        struct_to_fasta=_norm(stf),
        fasta_to_struct=_norm(fts),
        id=payload.get("id"),
        chain=payload.get("chain"),
        structure_path=payload.get("structure_path"),
        fasta_path=payload.get("fasta_path"),
    )


def resolve_map_json_from_structure(
    struct_path: Path,
    struct_root: Path,
    map_root: Path,
) -> Path:
    """Mirror the logic in biotite_mapping.py to find the correct `.map.json`.

    biotite_mapping.py writes:
      rel = struct_path.relative_to(struct_root)
      base = map_root / rel (drop suffix)
      out_json = base + '.map.json'

    So we replicate exactly.
    """
    rel = struct_path.relative_to(struct_root)
    base = map_root.joinpath(rel).with_suffix("")
    map_json = base.with_suffix(".map.json")
    return map_json


def _as_zero_based(idx: int, one_based: bool) -> int:
    return idx - 1 if one_based else idx


def _pretty_idx(idx0: Optional[int]) -> str:
    if idx0 is None:
        return "None"
    return f"{idx0} (1-based {idx0+1})"


def main():
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--map-json", type=Path, default=None, help="Path to a .map.json produced by biotite_mapping.py")
    src.add_argument("--struct-path", type=Path, default=None, help="Structure path (CIF/PDB) to auto-resolve its .map.json")

    ap.add_argument("--struct-root", type=Path, default=Path("structures"), help="Root folder containing CIF/PDB files")
    ap.add_argument("--map-root", type=Path, default=Path("interpretation/mappings"), help="Root folder containing .map.json files")

    ap.add_argument("--struct-idx", type=int, default=None, help="Index in seq_struct (0-based unless --one-based)")
    ap.add_argument("--fasta-idx", type=int, default=None, help="Index in FASTA (0-based unless --one-based)")

    ap.add_argument("--one-based", action="store_true", help="Interpret input indices as 1-based")

    args = ap.parse_args()

    if (args.struct_idx is None) == (args.fasta_idx is None):
        raise SystemExit("Provide exactly one of --struct-idx or --fasta-idx")

    # Locate mapping JSON
    if args.map_json is not None:
        map_json = args.map_json
    else:
        map_json = resolve_map_json_from_structure(args.struct_path, args.struct_root, args.map_root)

    if not map_json.exists():
        raise SystemExit(f"Mapping JSON not found: {map_json}")

    m = load_map_json(map_json)

    # Translate
    if args.struct_idx is not None:
        s0 = _as_zero_based(int(args.struct_idx), args.one_based)
        f0 = m.struct_idx_to_fasta_idx(s0)
        print("\n[STRUCT -> FASTA]")
        print(f"map_json: {m.map_json}")
        if m.structure_path:
            print(f"structure_path: {m.structure_path}")
        if m.fasta_path:
            print(f"fasta_path: {m.fasta_path}")
        if m.id:
            print(f"id: {m.id}  chain: {m.chain}")
        print(f"struct_idx: {_pretty_idx(s0)}")
        print(f"fasta_idx:  {_pretty_idx(f0)}")

    else:
        f0_in = _as_zero_based(int(args.fasta_idx), args.one_based)
        s0_out = m.fasta_idx_to_struct_idx(f0_in)
        print("\n[FASTA -> STRUCT]")
        print(f"map_json: {m.map_json}")
        if m.structure_path:
            print(f"structure_path: {m.structure_path}")
        if m.fasta_path:
            print(f"fasta_path: {m.fasta_path}")
        if m.id:
            print(f"id: {m.id}  chain: {m.chain}")
        print(f"fasta_idx:  {_pretty_idx(f0_in)}")
        print(f"struct_idx: {_pretty_idx(s0_out)}")


if __name__ == "__main__":
    main()