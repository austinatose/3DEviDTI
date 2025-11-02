import re
import pandas as pd

input_path = "out/pairs_filtered.csv"
output_path = "drug/drug_targets.csv"

df = pd.read_csv(input_path)
# name-structure pair


# ensure unique drugs
df = df.drop_duplicates(subset=['drug_name', 'smiles'])
print(f"Number of unique drugs found: {len(df)}")
df[['drug_name', 'smiles']].to_csv(output_path, index=False)