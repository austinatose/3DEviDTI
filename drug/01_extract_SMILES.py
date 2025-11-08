import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

input_path = "out/pairs_filtered.csv"
output_path = "drug/drug_targets.csv"

df = pd.read_csv(input_path)
# name-structure pair


# ensure unique drugs
df = df.drop_duplicates(subset=['drugbank_id', 'drug_name', 'smiles'])
print(f"Number of unique drugs found: {len(df)}")
# rename smiles column as SMILES

# prune those that RDKit cannot parse
valid_smiles = []
for i, smi in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_smiles.append(True)
    else:
        valid_smiles.append(False)

# export only valid ones
df = df[valid_smiles].reset_index(drop=True)
print(f"Number of valid SMILES: {len(df)}")

df = df.rename(columns={'smiles': 'SMILES'})
df[['drugbank_id', 'drug_name', 'SMILES']].to_csv(output_path, index=False)
