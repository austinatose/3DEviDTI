import re
import pandas as pd

input_path = "out/pairs_filtered.csv"
output_path = "targets.csv"

df = pd.read_csv(input_path)
unique_targets = df['target_uniprot'].dropna().unique()
unique_targets = [t for t in unique_targets if t != '']
print(f"Number of unique UniProt IDs found: {len(unique_targets)}")
pd.Series(unique_targets).to_csv(output_path, index=False, header=False)
