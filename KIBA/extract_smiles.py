
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Input: KIBA-style file where each non-empty line looks like:
# CHEMBL1087421 O00141 <SMILES> <AMINO_ACID_SEQUENCE> <VALUE>
#
# Example:
# CHEMBL1087421 O00141 COC1=C(...) 11.1
#
# We will extract: drug_id, uniprot_id, smiles, sequence (if present), value.

input_path = "data/Davis-KIBA/kiba.txt"  # adjust if needed
output_path = "lists/KIBA/KIBA_pairs.csv"

def parse_kiba_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()

    drug_id = parts[0]
    uniprot_id = parts[1]
    smiles = parts[2]
    sequence = parts[3]
    value = parts[4]

    # Try to convert value to float if possible
    value = float(value)

    if value >= 12.1:
        interaction = 1
    else:
        interaction = 0

    if len(smiles) > 200 or len(sequence) > 700:
        print(f"Skipping record with long SMILES or sequence: {drug_id}, {uniprot_id}")
        return None
    
    # parse inorganic molecules as well
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES, skipping: {smiles}")
            return None
    except Exception as e:
        print(f"Error parsing SMILES '{smiles}': {e}")
        return None

    return {
        "uniprot_id": uniprot_id,
        "drug_id": drug_id,
        
        "smiles": smiles,
        "Target sequence": sequence,
        # "kiba_value": value,
        "interaction": interaction
    }


def main():
    records = []

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        for line in f:
            rec = parse_kiba_line(line)
            if rec is None:
                continue
            records.append(rec)

    if not records:
        print("No valid records found. Nothing to write.")
        return

    df = pd.DataFrame(records)

    # Attach RDKit molecule column if you want to inspect in a notebook
    # PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles", molCol="ROMol", includeFingerprints=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} records with SMILES to: {output_path}")


if __name__ == "__main__":
    main()

