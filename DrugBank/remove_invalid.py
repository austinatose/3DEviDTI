import pandas as pd

INPUT_CSV = "lists/pairs_filtered.csv"
OUTPUT_CSV = "lists/pairs_filtered_newer.csv"

INVALIDS = ["unknown"]

df = pd.read_csv(INPUT_CSV)

mask = df['actions'].str.lower().apply(
    lambda x: not any(bad in str(x).lower() for bad in INVALIDS)
)
mask2 = df['known_action'].str.lower().apply(
    lambda x: not any(bad in str(x).lower() for bad in INVALIDS)
)
df_filtered = df[mask & mask2]

df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Removed {len(df) - len(df_filtered)} invalid rows. Saved to {OUTPUT_CSV}.")