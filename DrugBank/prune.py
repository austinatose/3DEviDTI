
#!/usr/bin/env python3
"""
DrugBank XML → (drug, target) pairs extractor with filtering.

Outputs two CSVs by default:
  1) pairs_raw.csv            — all parsed (drug, target) pairs with metadata
  2) pairs_filtered.csv       — after filtering out inorganics / very-small molecules

Filtering heuristics (tunable via CLI flags):
  • Inorganic: SMILES lacks any carbon atom token ('C' or 'c').
  • Very small: molecular weight (MW) < --mw-min (default: 100.0 Da).

Examples
--------
python prune.py full_database.xml \
  --out-dir out \
  --mw-min 120 \
  --keep-biotech false

Notes
-----
• Works in a streaming way (iterparse) and is memory-safe for full DrugBank XML.
• Robust to namespace variations; auto-detects default XML namespace.
• Captures UniProt accessions from external-identifiers where resource contains
  any of: 'UniProt', 'Swiss-Prot', 'TrEMBL'.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
from typing import Dict, Iterable, Optional, Tuple
import lxml.etree as ET
import tqdm

UNI_LIKE = ("UniProtKB", "UniProt", "Swiss-Prot", "TrEMBL", "UniProt Accession")

# ----------------------------
# Helpers
# ----------------------------

def detect_default_ns(xml_path: str) -> str:
    """Peek the root element to detect the default namespace URI (or empty string)."""
    for event, elem in ET.iterparse(xml_path, events=("start",)):
        tag = elem.tag
        # tag may look like '{namespace}drugbank' or 'drugbank' if no ns
        if tag.startswith("{"):
            ns_uri = tag.split("}", 1)[0][1:]
        else:
            ns_uri = ""
        # Clean up the parser
        elem.clear()
        return ns_uri
    return ""


def text(node: Optional[ET._Element]) -> Optional[str]:
    return None if node is None else (node.text or None)


def first(iterable: Iterable[str]) -> Optional[str]:
    for x in iterable:
        if x:
            return x
    return None


def is_inorganic_smiles(smiles: Optional[str]) -> bool:
    """Heuristic: inorganic if no carbon token appears in SMILES.
    Treat empty SMILES as inorganic=False (we don't want to drop solely for missing field).
    """
    if not smiles:
        return False
    return not bool(re.search(r"[Cc]", smiles))


def parse_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        # Some DrugBank weights include units or commas; strip non-numeric except dot
        s2 = re.sub(r"[^0-9.]+", "", s)
        try:
            return float(s2) if s2 else None
        except Exception:
            return None


# ----------------------------
# Core extraction
# ----------------------------

def extract_pairs(
    xml_path: str,
    out_raw_csv: str,
    out_filtered_csv: str,
    mw_min: float = 100.0,
    keep_biotech: bool = False,
    keep_inorganics: bool = False,
) -> Tuple[int, int, int, int]:
    """Stream-parse DrugBank XML and write raw and filtered pair CSVs.

    Returns tuple of counts: (n_drugs, n_pairs_raw, n_pairs_filtered, n_unique_targets)
    """
    ns_uri = detect_default_ns(xml_path)
    NS = {"db": ns_uri} if ns_uri else None

    def q(elem: ET._Element, path: str) -> Optional[ET._Element]:
        if NS:
            return elem.find(path, namespaces=NS)
        # Fallback: strip 'db:' and search without ns
        return elem.find(path.replace("db:", ""))

    def qall(elem: ET._Element, path: str) -> Iterable[ET._Element]:
        if NS:
            return elem.findall(path, namespaces=NS)
        return elem.findall(path.replace("db:", ""))

    drug_tag = f"{{{ns_uri}}}drug" if ns_uri else "drug"

    # Prepare writers
    os.makedirs(os.path.dirname(out_raw_csv) or ".", exist_ok=True)
    raw_f = open(out_raw_csv, "w", newline="")
    fil_f = open(out_filtered_csv, "w", newline="")
    raw_w = csv.writer(raw_f)
    fil_w = csv.writer(fil_f)

    header = [
        "drugbank_id",
        "drug_name",
        "drug_type",           # attribute on <drug type="small molecule|biotech|...">
        "groups",              # semicolon-joined drug groups (approved, investigational, etc.)
        "smiles",
        "inchikey",
        "molecular_weight",
        "target_uniprot",
        "target_id",
        "target_name",
        "target_organism",
        "actions",             # semicolon-joined
        "known_action",        # yes/no if present
    ]
    raw_w.writerow(header)
    fil_w.writerow(header)

    n_drugs = 0
    n_pairs_raw = 0
    n_pairs_filtered = 0
    uniq_targets = set()
    match_counts = {k: 0 for k in ("UniProtKB","UniProt","Swiss-Prot","TrEMBL","other")}  # debug

    # Stream over <drug> elements
    context = ET.iterparse(xml_path, events=("end",), tag=drug_tag)
    for _, drug in tqdm.tqdm(context):
        n_drugs += 1
        # Basic drug-level fields
        drug_type = drug.get("type") or ""
        dbid = None
        for dbid_el in qall(drug, ".//db:drugbank-id"):
            # Choose primary if annotated
            if dbid_el.get("primary") == "true":
                dbid = text(dbid_el)
                break
            if dbid is None:
                dbid = text(dbid_el)
        name = text(q(drug, "db:name")) or ""

        # Groups
        groups = ";".join([text(g) or "" for g in qall(drug, "db:groups/db:group") if text(g)])

        # Identifiers / properties
        smiles = None
        inchikey = None
        mw = None

        # Calculated properties often include SMILES, InChIKey, Molecular Weight
        for prop in qall(drug, "db:calculated-properties/db:property"):
            kind = text(q(prop, "db:kind")) or ""
            value = text(q(prop, "db:value")) or ""
            lk = kind.lower()
            if "smiles" in lk and not smiles:
                smiles = value.strip()
            elif "inchikey" in lk and not inchikey:
                inchikey = value.strip()
            elif "molecular weight" in lk and mw is None:
                mw = parse_float(value)

        # Targets
        for targ in qall(drug, "db:targets/db:target"):
            # Prefer UniProt at polypeptide level per XSD; fallback to target-level identifiers
            uniprot = None

            # 1) Polypeptide-level: use @id when source mentions UniProt; else check polypeptide/external-identifiers
            for pep in qall(targ, "db:polypeptide"):
                pep_source = (pep.get("source") or "")
                pep_id_attr = (pep.get("id") or "")
                if pep_id_attr and ("uniprot" in pep_source.lower()):
                    uniprot = pep_id_attr
                    break
                for ext in qall(pep, "db:external-identifiers/db:external-identifier"):
                    resource = (text(q(ext, "db:resource")) or "").strip()
                    ident    = (text(q(ext, "db:identifier")) or "").strip()
                    if ident and any(k.lower() in (resource or "").lower() for k in UNI_LIKE):
                        uniprot = ident
                        # debug tally
                        key = next((k for k in UNI_LIKE if k.lower() in resource.lower()), "other")
                        match_counts[key] = match_counts.get(key, 0) + 1
                        break
                if uniprot:
                    break

            # 2) Fallback: target-level external-identifiers (legacy)
            if not uniprot:
                for ext in qall(targ, "db:external-identifiers/db:external-identifier"):
                    resource = (text(q(ext, "db:resource")) or "").strip()
                    ident    = (text(q(ext, "db:identifier")) or "").strip()
                    if resource and ident:
                        rlow = resource.lower()
                        if any(k.lower() in rlow for k in UNI_LIKE):
                            uniprot = ident
                            # debug tally
                            key = next((k for k in UNI_LIKE if k.lower() in rlow), "other")
                            match_counts[key] = match_counts.get(key, 0) + 1
                            break

            # Target metadata
            t_id = text(q(targ, "db:id")) or ""
            tname = text(q(targ, "db:name")) or ""
            organism = text(q(targ, "db:organism")) or ""

            actions = ";".join([text(a) or "" for a in qall(targ, "db:actions/db:action") if text(a)])
            known_action = text(q(targ, "db:known-action")) or ""

            row = [dbid, name, drug_type, groups, smiles or "", inchikey or "",
                   mw if mw is not None else "", uniprot or "", t_id, tname, organism,
                   actions, known_action]
            raw_w.writerow(row)
            n_pairs_raw += 1

            # Filtering
            drop = False
            # Drop biotech class unless explicitly kept
            if not keep_biotech and drug_type.strip().lower() == "biotech":
                drop = True
            # Inorganic filter
            if not keep_inorganics and is_inorganic_smiles(smiles):
                drop = True
            # Very small molecule filter by MW
            if mw is not None and mw < mw_min:
                drop = True
            if smiles is None or smiles.strip() == "":
                drop = True

            # if not actions:
            #     drop = True
            # if actions.strip().lower() == "cofactor":
            #     drop = True
            if known_action.strip().lower() == "no":
                drop = True

            if not drop:
                fil_w.writerow(row)
                n_pairs_filtered += 1
                if uniprot:
                    uniq_targets.add(uniprot)

        # Free memory
        drug.clear()

    raw_f.close()
    fil_f.close()

    return n_drugs, n_pairs_raw, n_pairs_filtered, len(uniq_targets)


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Extract DrugBank (drug, target) pairs with filters")
    p.add_argument("xml", help="Path to DrugBank full_database.xml")
    p.add_argument("--out-dir", default="out", help="Directory to write CSV outputs")
    p.add_argument("--mw-min", type=float, default=100.0, help="Minimum molecular weight (Da) to keep")
    p.add_argument("--keep-biotech", type=lambda s: s.lower() == "true", default=False,
                   help="Keep biotech drugs (default: false)")
    p.add_argument("--keep-inorganics", type=lambda s: s.lower() == "true", default=False,
                   help="Keep inorganic compounds (default: false)")
    args = p.parse_args()

    raw_csv = os.path.join(args.out_dir, "pairs_raw.csv")
    fil_csv = os.path.join(args.out_dir, "pairs_filtered.csv")

    n_drugs, n_pairs_raw, n_pairs_filtered, n_uniq_t = extract_pairs(
        args.xml, raw_csv, fil_csv,
        mw_min=args.mw_min,
        keep_biotech=args.keep_biotech,
        keep_inorganics=args.keep_inorganics,
    )

    print("\nExtraction complete")
    print(f"Drugs parsed:            {n_drugs}")
    print(f"Pairs (raw):             {n_pairs_raw}")
    print(f"Pairs (filtered):        {n_pairs_filtered}")
    print(f"Unique UniProt targets:  {n_uniq_t}")
    print(f"Raw CSV:      {raw_csv}")
    print(f"Filtered CSV: {fil_csv}")
    # Note: match_counts is local to extract_pairs; to surface it here we'd typically return it.
    # For now, no extra print — keep CLI output minimal.


if __name__ == "__main__":
    main()