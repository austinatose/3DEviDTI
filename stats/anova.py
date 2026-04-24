from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import numpy as np

# Your 5 runs x 4 models
data = {
    'MolTrans':  [...],
    'DLM-DTI':   [...],
    'MocFormer': [...],
    '3DICE':     [...]
}

# ANOVA
f, p = stats.f_oneway(*data.values())
print(f"ANOVA p = {p:.4f}")

# Tukey's HSD
all_scores = np.concatenate(list(data.values()))
groups = np.repeat(list(data.keys()), 5)
print(pairwise_tukeyhsd(all_scores, groups))
