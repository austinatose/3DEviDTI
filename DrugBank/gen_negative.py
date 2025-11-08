import pandas as pd
import numpy as np
import random
import hashlib

PAIRS_CSV = "lists/pairs_valid.csv"
OUTPUT_CSV = "lists/pairs_final.csv"

with open(PAIRS_CSV, "rb") as f:
    print(hashlib.md5(f.read()).hexdigest())

valid_pairs_df = pd.read_csv(PAIRS_CSV)
positive_set = set(
    zip(valid_pairs_df['drugbank_id'], valid_pairs_df['uniprot_id'], np.ones(len(valid_pairs_df), dtype=bool))
)

negative_set = set()

while len(negative_set) < len(positive_set):
    drug = random.choice(list(positive_set))
    target = random.choice(list(positive_set))
    if drug[0] != target[1]:
        neg_pair = (drug[0], target[1], False)
        if neg_pair not in positive_set and neg_pair not in negative_set:
            negative_set.add(neg_pair)

final_set = positive_set.union(negative_set)
final_list = list(final_set)
random.shuffle(final_list)
final_df = pd.DataFrame(final_list, columns=['drugbank_id', 'uniprot_id', 'interaction'])
final_df.to_csv(OUTPUT_CSV, index=False)

with open(OUTPUT_CSV, "rb") as f:
    print(hashlib.md5(f.read()).hexdigest())
