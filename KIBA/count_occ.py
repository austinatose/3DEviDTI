import pandas as pd

path = "lists/KIBA/KIBA_pairs.csv"  # <- change this
df = pd.read_csv(path)

n_distinct = df["uniprot_id"].nunique(dropna=True)
print("Distinct drugs (by drug_id):", n_distinct)