import pandas as pd

DATA_PATH = 'lists/clean_val.csv'

df = pd.read_csv(DATA_PATH)
pos, neg = 0, 0

for index, row in df.iterrows():
    if row['interaction'] == 1:
        pos += 1
    else:
        neg += 1
print(pos,neg)
