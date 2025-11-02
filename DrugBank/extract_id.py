# # extract_id.py
# import lxml.etree as ET
# import tqdm

# # Path to your DrugBank XML file
# xml_path = "data/DrugBank/full_database.xml"
# output_path = "targets_uniprot.txt"

# ns = {"db": "http://www.drugbank.ca"}  # confirm namespace from your XML header

# uniprot_ids = set()  # use set to avoid duplicates

# for event, elem in tqdm.tqdm(ET.iterparse(xml_path, events=("end",), tag="{http://www.drugbank.ca}target")):
#     for ext_id in elem.findall(".//db:external-identifiers/db:external-identifier", namespaces=ns):
#         resource = ext_id.findtext("db:resource", namespaces=ns)
#         identifier = ext_id.findtext("db:identifier", namespaces=ns)
#         if resource and "UniProtKB" in resource:
#             uniprot_ids.add(identifier)
#     elem.clear()

# # Write all IDs to file
# with open(output_path, "w") as f:
#     for uid in sorted(uniprot_ids):
#         f.write(uid + "\n")

# print(f"Extracted {len(uniprot_ids)} unique UniProt IDs to {output_path}")

# extract_id.py — only UniProt IDs of proteins that are **targets** of drugs
import lxml.etree as ET
import tqdm

xml_path = "data/DrugBank/full_database.xml"
output_path = "targets_uniprot.txt"

# Confirm the namespace from your XML header
NS = {"db": "http://www.drugbank.ca"}

# Optional filters
REQUIRE_ANY_ACTION = False   # set True to require at least one action term
REQUIRE_KNOWN_ACTION_YES = False  # set True to require <known-action>yes</known-action>

UNI_LIKE = ("UniProtKB", "UniProt", "Swiss-Prot", "TrEMBL", "UniProt Accession")

uniprot_ids = set()

def has_uniprot_id(target_elem):
    """Return a UniProt accession for this <target>, preferring polypeptide-level IDs.
    Order of preference:
      1) <polypeptide @id> when polypeptide@source mentions UniProt
      2) <polypeptide>/<external-identifiers> with resource in UNI_LIKE
      3) legacy fallback: <target>/<external-identifiers>
    """
    # 1) Prefer polypeptide@id when source mentions UniProt
    for pep in target_elem.findall("db:polypeptide", namespaces=NS):
        pep_source = (pep.get("source") or "")
        pep_id_attr = (pep.get("id") or "")
        if pep_id_attr and ("uniprot" in pep_source.lower()):
            return pep_id_attr
        # 2) Else check polypeptide/external-identifiers
        for ext in pep.findall("db:external-identifiers/db:external-identifier", namespaces=NS):
            resource = (ext.findtext("db:resource", namespaces=NS) or "").strip()
            ident    = (ext.findtext("db:identifier", namespaces=NS) or "").strip()
            if ident and any(x.lower() in resource.lower() for x in UNI_LIKE):
                return ident

    # 3) Fallback: target-level external-identifiers (legacy)
    for ext in target_elem.findall("db:external-identifiers/db:external-identifier", namespaces=NS):
        resource = (ext.findtext("db:resource", namespaces=NS) or "").strip()
        ident    = (ext.findtext("db:identifier", namespaces=NS) or "").strip()
        if ident and any(x.lower() in resource.lower() for x in UNI_LIKE):
            return ident

    return None

def passes_action_filters(target_elem):
    if REQUIRE_ANY_ACTION:
        actions = target_elem.findall("db:actions/db:action", namespaces=NS)
        if not actions:
            return False
    if REQUIRE_KNOWN_ACTION_YES:
        known = (target_elem.findtext("db:known-action", namespaces=NS) or "").strip().lower()
        if known != "yes":
            return False
    return True

# Stream by <drug> and then **only** traverse drug/targets/target (this avoids enzymes/carriers/transporters)
for _, drug in tqdm.tqdm(ET.iterparse(xml_path, events=("end",), tag="{http://www.drugbank.ca}drug")):
    for targ in drug.findall("db:targets/db:target", namespaces=NS):
        if not passes_action_filters(targ):
            continue
        acc = has_uniprot_id(targ)
        if acc:
            uniprot_ids.add(acc)
    drug.clear()

with open(output_path, "w") as f:
    for uid in sorted(uniprot_ids):
        f.write(uid + "\n")

print(f"Extracted {len(uniprot_ids)} unique UniProt target IDs to {output_path}")