"""One-shot extraction of drug + protein names for the search endpoint.

Outputs (under `webapp/.cache/`):
  drug_names.json:    dict[drug_id] -> {"name": str, "aliases": list[str]}
  protein_names.json: dict[uniprot_id] -> {"name": str, "aliases": list[str]}

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


def extract_drug_names() -> dict[str, dict]:
    """Stream-parse DrugBank XML for primary id, display name, and synonyms.

    Crucial subtlety: the XML nests `<salt>` blocks (with their own `<drugbank-id primary="true">`)
    and `<polypeptide>` blocks (with their own `<name>`) inside each `<drug>`. We must only treat
    elements that are *direct children* of the outermost `<drug>` as the drug's metadata,
    otherwise salt IDs overwrite the parent's primary ID and we lose ~1.6k drugs.
    """
    if not DRUGBANK_XML.exists():
        print(f"[indices] skipping drug names: {DRUGBANK_XML} missing")
        return {}

    out: dict[str, dict] = {}
    path: list[str] = []
    current_id: str | None = None
    current_name: str | None = None
    current_aliases: list[str] = []

    # Path patterns we treat as alias sources (each ends in `<name>` or `<synonym>` text).
    # The XML root is <drugbank>, so the path looks like ['drugbank', 'drug', ...]. We compute
    # the descendant path relative to the outermost <drug> element.
    #   drug/synonyms/synonym                       — INN/regulator names, e.g. "Leuprorelin"
    #   drug/international-brands/international-brand/name
    #   drug/products/product/name                  — e.g. "Celebrex"
    def _alias_match() -> bool:
        if path.count("drug") != 1 or "drug" not in path:
            return False
        sub = path[path.index("drug"):]
        if len(sub) == 3 and sub[1] == "synonyms" and sub[2] == "synonym":
            return True
        if len(sub) == 4 and sub[1] in ("international-brands", "products") and sub[3] == "name":
            return True
        return False

    context = iterparse(str(DRUGBANK_XML), events=("start", "end"))
    for event, elem in context:
        tag = elem.tag.split("}")[-1]
        if event == "start":
            path.append(tag)
            continue

        # event == "end"
        parent = path[-2] if len(path) >= 2 else None
        # `path.count("drug") == 1` ensures we ignore nested `<drug>`s (e.g., in interactions).
        is_top_drug_child = parent == "drug" and path.count("drug") == 1

        if tag == "drugbank-id" and is_top_drug_child and elem.attrib.get("primary") == "true":
            current_id = (elem.text or "").strip()
        elif tag == "name" and is_top_drug_child and current_name is None:
            current_name = (elem.text or "").strip()
        elif (tag == "synonym" or tag == "name") and _alias_match():
            txt = (elem.text or "").strip()
            if txt:
                current_aliases.append(txt)
        elif tag == "drug" and path.count("drug") == 1:
            if current_id and current_name:
                out[current_id] = {
                    "name": current_name,
                    "aliases": list(dict.fromkeys(current_aliases)),  # dedupe, keep order
                }
            current_id = None
            current_name = None
            current_aliases = []
            elem.clear()

        path.pop()

    print(f"[indices] parsed {len(out)} drug names")
    return out


# UniProt FASTA header:  >sp|P35354|PGH2_HUMAN Prostaglandin G/H synthase 2 OS=Homo sapiens OX=9606 GN=PTGS2 PE=1 SV=2
_FASTA_HEADER = re.compile(
    r"^>(?:sp|tr)\|(?P<id>[^|]+)\|(?P<entry>\S+)\s+(?P<name>.+?)(?:\s+OS=|\s*$)"
)
_GN_FIELD = re.compile(r"\sGN=([^\s]+)")


def extract_protein_names() -> dict[str, dict]:
    if not FASTA_DIR.exists():
        print(f"[indices] skipping protein names: {FASTA_DIR} missing")
        return {}
    out: dict[str, dict] = {}
    for fasta in FASTA_DIR.glob("*.fasta"):
        try:
            first = fasta.read_text().splitlines()[0]
        except Exception:
            continue
        m = _FASTA_HEADER.match(first)
        if not m:
            continue
        aliases: list[str] = []
        # The entry name (e.g. PGH2_HUMAN) and gene symbol (e.g. PTGS2) are common search terms.
        entry = m.group("entry").strip()
        if entry:
            aliases.append(entry)
            short = entry.split("_", 1)[0]  # PGH2_HUMAN -> PGH2
            if short and short != entry:
                aliases.append(short)
        gn = _GN_FIELD.search(first)
        if gn:
            aliases.append(gn.group(1).strip())
        out[m.group("id")] = {
            "name": m.group("name").strip(),
            "aliases": list(dict.fromkeys(aliases)),
        }
    print(f"[indices] parsed {len(out)} protein names")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "drug_names.json").write_text(json.dumps(extract_drug_names()))
    (OUT_DIR / "protein_names.json").write_text(json.dumps(extract_protein_names()))
    print(f"[indices] wrote {OUT_DIR}/drug_names.json and protein_names.json")


if __name__ == "__main__":
    main()
