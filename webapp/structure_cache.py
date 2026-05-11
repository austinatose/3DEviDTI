"""Fetch + cache PDB structures and align embedding indices to PDB author residue numbers."""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard residues canonicalised to nearest parent.
    "MSE": "M", "SEC": "C", "PYL": "K", "SEP": "S", "TPO": "T", "PTR": "Y",
    "CSO": "C", "CSD": "C", "CME": "C", "CYX": "C", "CSS": "C",
    "HIP": "H", "HIE": "H", "HID": "H", "ASH": "D", "GLH": "E",
    "MHO": "M", "HYP": "P",
}


@dataclass
class ProteinStructure:
    uniprot_id: str
    pdb_id: str
    source: str
    chain: str
    pdb_text: str  # PDB-format text suitable for 3Dmol.js
    # Each entry corresponds to one ESM-IF1 struct_idx. resnums[i] is None when the
    # mapping JSON's seq_struct claims a residue that the PDB doesn't model (gap).
    resnums: list[Optional[int]]
    icodes: list[str]
    seq_struct: str  # the mapping JSON's seq_struct, unchanged (length L_struct)


def _parse_struct_path(struct_path: str) -> tuple[str, str, str]:
    """Return (structure_id, chain, source). Handles PDB-style `5F19_A.cif` and AlphaFold-style
    `AF-A0A023W3H0-F1-model.cif` (chain assumed to be 'A')."""
    name = Path(struct_path).stem
    m_pdb = re.match(r"^([0-9A-Za-z]{4})_([A-Za-z0-9])$", name)
    if m_pdb:
        return m_pdb.group(1).upper(), m_pdb.group(2), "PDB"
    m_af = re.match(r"^AF-([A-Z0-9]+)-F\d+-model.*$", name)
    if m_af:
        return f"AF-{m_af.group(1)}", "A", "AlphaFold"
    raise ValueError(f"could not parse structure id/chain from {struct_path!r}")


def _fetch_pdb(structure_id: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{structure_id}.pdb"
    if path.exists():
        return path.read_text()
    if structure_id.startswith("AF-"):
        uniprot = structure_id[len("AF-"):]
        # AlphaFold's file versioning bumps periodically (v4 → v6 → …); use their prediction
        # API to discover the current pdbUrl rather than hard-coding the version.
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        try:
            with urllib.request.urlopen(api_url, timeout=20) as r:
                entries = json.loads(r.read().decode())
        except Exception as e:
            raise RuntimeError(f"AlphaFold API lookup failed for {structure_id}: {e}")
        if not entries or "pdbUrl" not in entries[0]:
            raise RuntimeError(f"AlphaFold has no model for {uniprot}")
        pdb_url = entries[0]["pdbUrl"]
        with urllib.request.urlopen(pdb_url, timeout=30) as r:
            path.write_bytes(r.read())
        return path.read_text()
    url = f"https://files.rcsb.org/download/{structure_id}.pdb"
    with urllib.request.urlopen(url, timeout=20) as r:
        path.write_bytes(r.read())
    return path.read_text()


def pdb_text_for_chain(pdb_text: str, chain: str) -> str:
    """Return PDB coordinate records for only one chain."""
    coord_records = ("ATOM  ", "HETATM", "ANISOU", "TER   ")
    lines = [
        line
        for line in pdb_text.splitlines()
        if line.startswith(coord_records) and len(line) > 21 and line[21] == chain
    ]
    lines.append("END")
    return "\n".join(lines) + "\n"


def _residues_with_ca(pdb_text: str, chain: str) -> list[tuple[int, str, str]]:
    """Return (auth_seq_id, icode, resname_3) for each chain residue that has a CA atom,
    in PDB order, in the first alt conf. Accepts both ATOM and HETATM records — needed for
    non-standard residues like MSE that ESM-IF1's `load_coords` includes via the canonical map."""
    seen: set[tuple[int, str]] = set()
    out: list[tuple[int, str, str]] = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 27:
            continue
        if line[21] != chain:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resname = line[17:20].strip().upper()
        if line.startswith("HETATM") and resname not in THREE_TO_ONE:
            continue
        alt_loc = line[16] if len(line) > 16 else " "
        if alt_loc not in (" ", "A"):
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26] if len(line) > 26 else " "
        icode = "" if icode == " " else icode
        key = (resnum, icode)
        if key in seen:
            continue
        seen.add(key)
        out.append((resnum, icode, resname))
    return out


def _align(target_seq: str, recovered_seq: str, recovered_meta: list[tuple[int, str, str]]):
    """Greedy 2-pointer alignment of `target_seq` (from mapping JSON) to `recovered_seq` (from PDB).

    Allows the PDB to be missing residues (output entry becomes None at that struct_idx) and
    allows the PDB to contain extras that don't appear in `target_seq` (we skip past them with a
    small lookahead). Returns parallel lists of length len(target_seq) of (resnum_or_None,
    icode_or_'') or raises RuntimeError if the sequences are too divergent to align cleanly.
    """
    LOOKAHEAD = 4
    resnums: list[Optional[int]] = []
    icodes: list[str] = []
    j = 0
    for i, t in enumerate(target_seq):
        if j < len(recovered_seq) and recovered_seq[j] == t:
            resnums.append(recovered_meta[j][0])
            icodes.append(recovered_meta[j][1])
            j += 1
            continue
        # Try skipping a few extras in `recovered` to find a match (PDB has residues not in seq_struct).
        skip = next(
            (
                k
                for k in range(1, LOOKAHEAD + 1)
                if j + k < len(recovered_seq) and recovered_seq[j + k] == t
            ),
            None,
        )
        if skip is not None:
            j += skip + 1
            resnums.append(recovered_meta[j - 1][0])
            icodes.append(recovered_meta[j - 1][1])
            continue
        # PDB is missing a residue that seq_struct expects here.
        resnums.append(None)
        icodes.append("")
    if j != len(recovered_seq):
        raise RuntimeError(
            f"alignment did not consume all PDB residues: stopped at j={j}/{len(recovered_seq)}"
        )
    return resnums, icodes


def load_structure(
    uniprot_id: str,
    map_json_path: Path,
    cache_dir: Path,
) -> ProteinStructure:
    payload = json.loads(map_json_path.read_text())
    pdb_id, chain, source = _parse_struct_path(payload["structure_path"])
    expected_seq = payload["seq_struct"]
    L_struct = int(payload["L_struct"])

    pdb_text = _fetch_pdb(pdb_id, cache_dir)
    residues = _residues_with_ca(pdb_text, chain)
    recovered_seq = "".join(THREE_TO_ONE.get(r[2], "X") for r in residues)

    resnums, icodes = _align(expected_seq, recovered_seq, residues)
    if len(resnums) != L_struct:
        raise RuntimeError(
            f"alignment produced {len(resnums)} entries, expected {L_struct}"
        )

    n_gaps = sum(1 for r in resnums if r is None)
    if n_gaps:
        print(
            f"[structure] {uniprot_id} ({pdb_id} chain {chain}): "
            f"{n_gaps} residue(s) in seq_struct have no PDB coords"
        )

    return ProteinStructure(
        uniprot_id=uniprot_id,
        pdb_id=pdb_id,
        source=source,
        chain=chain,
        pdb_text=pdb_text,
        resnums=resnums,
        icodes=icodes,
        seq_struct=expected_seq,
    )
