"""One-shot extraction of drug + protein names for the search endpoint.

Outputs (under `webapp/.cache/`):
  drug_names.json:    dict[drug_id] -> drug name (DrugBank `<name>` element)
  protein_names.json: dict[uniprot_id] -> protein name (UniProt FASTA description)

Re-run when the source files change. Idempotent — overwrites the JSONs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree.ElementTree import iterparse

ROOT = Path(__file__).resolve().parent.parent
DRUGBANK_XML = ROOT / "data" / "DrugBank" / "full_database.xml"
FASTA_DIR = ROOT / "other" / "uniprot_fasta"
OUT_DIR = Path(__file__).resolve().parent / ".cache"


def extract_drug_names() -> dict[str, str]:
    if not DRUGBANK_XML.exists():
        print(f"[indices] skipping drug names: {DRUGBANK_XML} missing")
        return {}
    out: dict[str, str] = {}
    # Stream-parse to avoid loading the 1.5 GB XML into memory.
    context = iterparse(str(DRUGBANK_XML), events=("end",))
    current_id: str | None = None
    current_name: str | None = None
    for _, elem in context:
        tag = elem.tag.split("}")[-1]
        if tag == "drugbank-id":
            if elem.attrib.get("primary") == "true":
                current_id = (elem.text or "").strip()
        elif tag == "name" and current_name is None and current_id:
            current_name = (elem.text or "").strip()
        elif tag == "drug":
            if current_id and current_name:
                out[current_id] = current_name
            current_id = None
            current_name = None
            elem.clear()
    print(f"[indices] parsed {len(out)} drug names")
    return out


_FASTA_HEADER = re.compile(r"^>(?:sp|tr)\|([^|]+)\|\S+\s+(.+?)(?:\s+OS=|\s*$)")


def extract_protein_names() -> dict[str, str]:
    if not FASTA_DIR.exists():
        print(f"[indices] skipping protein names: {FASTA_DIR} missing")
        return {}
    out: dict[str, str] = {}
    for fasta in FASTA_DIR.glob("*.fasta"):
        try:
            first = fasta.read_text().splitlines()[0]
        except Exception:
            continue
        m = _FASTA_HEADER.match(first)
        if not m:
            continue
        out[m.group(1)] = m.group(2).strip()
    print(f"[indices] parsed {len(out)} protein names")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "drug_names.json").write_text(json.dumps(extract_drug_names()))
    (OUT_DIR / "protein_names.json").write_text(json.dumps(extract_protein_names()))
    print(f"[indices] wrote {OUT_DIR}/drug_names.json and protein_names.json")


if __name__ == "__main__":
    main()
