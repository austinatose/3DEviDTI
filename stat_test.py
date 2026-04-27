
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# =====================================================================
# DATA: replace placeholder rows with your actual MolTrans / DLM-DTI runs
# Rows  = 5 replications
# Cols  = ACC, PPV, TPR, F1, MCC, AUROC, AUPRC
# =====================================================================

metrics = ['ACC', 'PPV', 'TPR', 'F1', 'MCC', 'AUROC', 'AUPRC']

# DrugBank

DICE = np.array([
    [0.8909946, 0.76837, 0.62623, 0.69006, 0.62957, 0.92367, 0.79608],
    [0.891435, 0.7636, 0.6368, 0.6945, 0.6330, 0.9242, 0.7943],
    [0.8909946, 0.73590, 0.68234, 0.70810, 0.64190, 0.92311, 0.79492],
    [0.8924636, 0.78303, 0.61562, 0.68930, 0.63202, 0.92258, 0.79243],
    [0.8915822, 0.77175, 0.62547, 0.70988, 0.64183, 0.92200, 0.79253]
])

MOC = np.array([
    [0.8915822, 0.739095, 0.680819, 0.708761, 0.643110, 0.92200, 0.791340],
    [0.889085, 0.736181, 0.666414, 0.699562, 0.632934, 0.922191, 0.791280],
    [0.889966, 0.741525, 0.663381, 0.699562, 0.634621, 0.921642, 0.790452],
    [0.890701, 0.725490, 0.701289, 0.713184, 0.643162, 0.922468, 0.791459],
    [0.890701, 0.733930, 0.683851, 0.708006, 0.641482, 0.922468, 0.791699]
])

MOLTRANS = np.array([
    [0.87926, 0.70763, 0.65845, 0.68215, 0.60835, 0.90530, 0.74515],
    [0.88088, 0.69751, 0.68058, 0.68894, 0.61536, 0.90999, 0.77300],
    [0.87971, 0.73009, 0.61659, 0.66856, 0.59889, 0.89903, 0.73276],
    [0.88235, 0.72319, 0.65071, 0.68504, 0.61425, 0.90603, 0.75546],
    [0.88471, 0.73617, 0.64600, 0.68815, 0.61978, 0.90291, 0.75010]
])

DLM_DTI = np.array([
    [0.86514, 0.65087, 0.6558, 0.65332, 0.56962 , 0.89167 , 0.72238],
    [0.86558, 0.73059, 0.48522, 0.58314, 0.52197, 0.87968, 0.68902],
    [0.87087, 0.70561, 0.5724, 0.63206, 0.55925, 0.88608, 0.70457],
    [0.87234, 0.74038, 0.5254, 0.61463, 0.55219, 0.88577, 0.70503],
    [0.86543, 0.6573, 0.63836, 0.64769, 0.56464, 0.89097, 0.7056]
])

models_data = {
    'MolTrans':  MOLTRANS,
    'DLM-DTI':   DLM_DTI,
    'MocFormer': MOC,
    '3DICE':     DICE,
}

REFERENCE = '3DICE'   # all post-hoc comparisons are baseline vs this model
ALPHA = 0.05
N_RUNS = 5
CORRECTION = 'holm'

# =====================================================================
# Helpers
# =====================================================================

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def has_data(arr):
    return not np.isnan(arr).any()

def direction(diff_ref_minus_baseline):
    """diff = 3DICE - baseline. Positive means 3DICE is better."""
    return 'better' if diff_ref_minus_baseline > 0 else 'worse'

# =====================================================================
# Per-metric analysis
# =====================================================================

active_models = {k: v for k, v in models_data.items() if has_data(v)}
baselines = {k: v for k, v in active_models.items() if k != REFERENCE}
ref_data = active_models[REFERENCE]

print("="*100)
print(f"REPEATED MEASURES ANOVA + POST-HOC: {REFERENCE} vs each baseline ({CORRECTION}-corrected)")
print(f"Models: {list(active_models.keys())}")
print(f"Replications per model: {N_RUNS}")
print("="*100)

for i, m in enumerate(metrics):
    print(f"\n{'='*100}")
    print(f"METRIC: {m}")
    print('='*100)

    # Mean ± std for all models
    print(f"  {'Model':<12} {'Mean':>10} {'Std':>10}")
    for name, data in active_models.items():
        col = data[:, i]
        marker = ' ◄' if name == REFERENCE else ''
        print(f"  {name:<12} {col.mean():>10.6f} {col.std(ddof=1):>10.6f}{marker}")

    # --- RM-ANOVA (omnibus across all 4 models) ---
    rows = []
    for name, data in active_models.items():
        for run_idx, val in enumerate(data[:, i]):
            rows.append({'subject': run_idx, 'model': name, 'score': val})
    df = pd.DataFrame(rows)

    aov = AnovaRM(df, depvar='score', subject='subject', within=['model']).fit()
    f_stat = aov.anova_table['F Value'].iloc[0]
    p_anova = aov.anova_table['Pr > F'].iloc[0]
    print(f"\n  RM-ANOVA (omnibus): F = {f_stat:.4f}, p = {p_anova:.6f}  {sig_stars(p_anova)}")

    if p_anova >= ALPHA:
        print(f"  → Omnibus test not significant. Skipping post-hoc.")
        continue

    # --- Post-hoc: 3DICE vs each baseline only ---
    # Correction applied across the 3 comparisons (not all 6 pairs)
    print(f"\n  Post-hoc: {REFERENCE} vs each baseline ({CORRECTION}-corrected across {len(baselines)} comparisons):")

    pairs, raw_ps, raw_ts, mean_diffs = [], [], [], []
    for baseline_name, baseline_data in baselines.items():
        ref_scores      = ref_data[:, i]
        baseline_scores = baseline_data[:, i]
        t, p = stats.ttest_rel(ref_scores, baseline_scores)
        pairs.append(baseline_name)
        raw_ts.append(t)
        raw_ps.append(p)
        mean_diffs.append(ref_scores.mean() - baseline_scores.mean())  # 3DICE - baseline

    reject, p_corr, _, _ = multipletests(raw_ps, alpha=ALPHA, method=CORRECTION)

    print(f"    {'Comparison':<30} {'3DICE−baseline':>15} {'t':>8} {'p (raw)':>10} {'p (adj)':>10} {'result':>16}")
    for baseline_name, md, t, p_raw, p_adj in zip(pairs, mean_diffs, raw_ts, raw_ps, p_corr):
        comparison = f"3DICE vs {baseline_name}"
        direc = direction(md)
        label = f"{sig_stars(p_adj)} ({direc})" if p_adj < ALPHA else 'ns'
        print(f"    {comparison:<30} {md:>+15.6f} {t:>8.3f} {p_raw:>10.6f} {p_adj:>10.6f} {label:>16}")

print(f"\n{'='*100}")
print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
print("3DICE−baseline > 0 means 3DICE is better than the baseline.")
print('='*100)