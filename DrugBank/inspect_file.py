import lxml.etree as ET
NS = {"db": "http://www.drugbank.ca"}

xml_path = "data/DrugBank/full_database.xml"

for _, drug in ET.iterparse(xml_path, events=("end",), tag="{http://www.drugbank.ca}drug"):
    # primary DrugBank ID
    dbid = None
    for el in drug.findall("db:drugbank-id", namespaces=NS):
        if el.get("primary") == "true":
            dbid = el.text
            break
        if dbid is None:
            dbid = el.text

    for targ in drug.findall("db:targets/db:target", namespaces=NS):
        target_id = targ.findtext("db:id", namespaces=NS) or ""
        target_name = targ.findtext("db:name", namespaces=NS) or ""
        organism   = targ.findtext("db:organism", namespaces=NS) or ""

        uniprot = None

        # 1) Prefer polypeptide@id when source mentions UniProt
        for pep in targ.findall("db:polypeptide", namespaces=NS):
            pep_source = (pep.get("source") or "")
            pep_id_attr = (pep.get("id") or "")
            if pep_id_attr and ("uniprot" in pep_source.lower()):
                uniprot = pep_id_attr
                break
            # 2) Else check polypeptide/external-identifiers
            for ext in pep.findall("db:external-identifiers/db:external-identifier", namespaces=NS):
                resource = (ext.findtext("db:resource", namespaces=NS) or "").strip()
                ident    = (ext.findtext("db:identifier", namespaces=NS) or "").strip()
                if ident and resource in ("UniProtKB", "UniProt Accession"):
                    uniprot = ident
                    break
            if uniprot:
                break

        # 3) Legacy fallback: target-level external-identifiers
        if not uniprot:
            for ext in targ.findall("db:external-identifiers/db:external-identifier", namespaces=NS):
                resource = (ext.findtext("db:resource", namespaces=NS) or "").strip()
                ident    = (ext.findtext("db:identifier", namespaces=NS) or "").strip()
                if ident and resource in ("UniProtKB", "UniProt", "Swiss-Prot", "TrEMBL", "UniProt Accession"):
                    uniprot = ident
                    break

        print(dbid, target_id, target_name, organism, uniprot)

    drug.clear()