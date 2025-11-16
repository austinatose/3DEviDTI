import pandas as pd
import numpy as np
import random
import hashlib
from pathlib import Path

PAIRS_CSV = "lists/only_clean_pairs.csv"
OUTPUT_CSV = "lists/clean_final.csv"
TRAIN_CSV = "lists/clean_train.csv"
VAL_CSV   = "lists/clean_val.csv"
TEST_CSV  = "lists/clean_test.csv"

with open(PAIRS_CSV, "rb") as f:
    print(hashlib.md5(f.read()).hexdigest())

valid_pairs_df = pd.read_csv(PAIRS_CSV)

# Build unique positive (drug, target) pairs
pos_pairs = list({(d, t) for d, t in zip(valid_pairs_df['drugbank_id'], valid_pairs_df['uniprot_id'])})
pos_set = set(pos_pairs)
print(f"Unique positive pairs: {len(pos_set)}")

pos_drugs   = [d for d, _ in pos_pairs]
pos_targets = [t for _, t in pos_pairs]

neg_set: set[tuple[str, str]] = set()
while len(neg_set) < len(pos_set):
    t = random.choice(pos_targets)
    d = random.choice(pos_drugs)
    if (d, t) not in pos_set and (d, t) not in neg_set:
        neg_set.add((d, t))

print(f"Generated negatives: {len(neg_set)}")

pos_df = pd.DataFrame(list(pos_set), columns=['drugbank_id', 'uniprot_id'])
pos_df['interaction'] = 1
neg_df = pd.DataFrame(list(neg_set), columns=['drugbank_id', 'uniprot_id'])
neg_df['interaction'] = 0

final_df = pd.concat([pos_df, neg_df], ignore_index=True)
final_df = final_df.sample(frac=1.0).reset_index(drop=True)

final_df.to_csv(OUTPUT_CSV, index=False)
with open(OUTPUT_CSV, "rb") as f:
    print(hashlib.md5(f.read()).hexdigest())

# might as well split here
# yay my dataset is nicely divisible by 10

def split_class(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    df = df.sample(frac=1.0).reset_index(drop=True)
    n = len(df)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test

pos_train, pos_val, pos_test = split_class(pos_df)
neg_train, neg_val, neg_test = split_class(neg_df)

print("Positive splits:", len(pos_train), len(pos_val), len(pos_test))
print("Negative splits:", len(neg_train), len(neg_val), len(neg_test))

train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1.0).reset_index(drop=True)
val_df = pd.concat([pos_val, neg_val], ignore_index=True).sample(frac=1.0).reset_index(drop=True)
test_df = pd.concat([pos_test,neg_test], ignore_index=True).sample(frac=1.0).reset_index(drop=True)

for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
    counts = df['interaction'].value_counts()
    print(f"{name}: total={len(df)}, pos={int(counts.get(1,0))}, neg={int(counts.get(0,0))}")

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    with open(p, 'rb') as f:
        print(Path(p).name, hashlib.md5(f.read()).hexdigest())
