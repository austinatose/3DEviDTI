
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

DICE = np.array([
    [0.810669, 0.7931, 0.8580, 0.8162, 0.6225, 0.8832, 0.8730],
    [0.810293, 0.7931, 0.8580, 0.8134, 0.6209, 0.8835, 0.8730],
    [0.806526, 0.7772, 0.8595, 0.8163, 0.6165, 0.8765, 0.8621],
    [0.808415, 0.7893, 0.8415, 0.8145, 0.6182, 0.8717, 0.8648],
    [0.809542, 0.7918, 0.8603, 0.8172, 0.6213, 0.8787, 0.8648],
])

MOC = np.array([
    [0.793764, 0.779301, 0.8542,   0.8126,   0.593673, 0.878163, 0.8653  ],
    [0.798272, 0.773040, 0.844478, 0.807181, 0.599108, 0.882604, 0.881273],
    [0.798648, 0.773196, 0.845229, 0.807609, 0.599904, 0.882562, 0.881240],
    [0.798648, 0.773572, 0.844478, 0.807471, 0.599820, 0.882643, 0.881391],
    [0.796394, 0.771134, 0.842975, 0.805456, 0.595377, 0.881884, 0.880532],
])

# TODO: replace these placeholder rows with your real MolTrans + DLM-DTI runs
MOLTRANS = np.array([
    [np.nan]*7,
    [np.nan]*7,
    [np.nan]*7,
    [np.nan]*7,
    [np.nan]*7,
])

DLM_DTI = np.array([
    [np.nan]*7,
    [np.nan]*7,
    [np.nan]*7,
    [np.nan]*7,
    [np.nan]*7,
])

models_data = {
    'MolTrans':  MOLTRANS,
    'DLM-DTI':   DLM_DTI,
    'MocFormer': MOC,
    '3DICE':     DICE,
}

ALPHA = 0.05
N_RUNS = 5
CORRECTION = 'holm'   # 'holm' or 'bonferroni'

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

# =====================================================================
# Per-metric analysis
# =====================================================================

print("="*100)
print(f"REPEATED MEASURES ANOVA + PAIRED POST-HOC TESTS (correction: {CORRECTION})")
print(f"Models: {list(models_data.keys())}")
print(f"Replications per model: {N_RUNS}")
print("="*100)

# Drop models with no real data
active_models = {k: v for k, v in models_data.items() if has_data(v)}
if len(active_models) < len(models_data):
    skipped = set(models_data) - set(active_models)
    print(f"\n[WARNING] Skipping models with placeholder NaN data: {skipped}")
    print(f"          Running with {len(active_models)} models: {list(active_models)}\n")

if len(active_models) < 2:
    print("\n[ERROR] Need at least 2 models with data. Fill in real values and re-run.")
    raise SystemExit

for i, m in enumerate(metrics):
    print(f"\n{'='*100}")
    print(f"METRIC: {m}")
    print('='*100)

    # Print mean ± std
    print(f"  {'Model':<12} {'Mean':>10} {'Std':>10}")
    for name, data in active_models.items():
        col = data[:, i]
        print(f"  {name:<12} {col.mean():>10.6f} {col.std(ddof=1):>10.6f}")

    # --- Repeated Measures ANOVA ---
    rows = []
    for name, data in active_models.items():
        for run_idx, val in enumerate(data[:, i]):
            rows.append({'subject': run_idx, 'model': name, 'score': val})
    df = pd.DataFrame(rows)

    aov = AnovaRM(df, depvar='score', subject='subject', within=['model']).fit()
    f_stat = aov.anova_table['F Value'].iloc[0]
    p_anova = aov.anova_table['Pr > F'].iloc[0]
    print(f"\n  RM-ANOVA: F = {f_stat:.4f}, p = {p_anova:.6f}  {sig_stars(p_anova)}")

    if p_anova >= ALPHA:
        print(f"  → Omnibus test not significant. Skipping post-hoc.")
        continue

    # --- Pairwise paired t-tests with multiple-comparison correction ---
    print(f"\n  Post-hoc paired t-tests ({CORRECTION}-corrected):")
    pairs, raw_ps, raw_ts, mean_diffs = [], [], [], []
    for a, b in combinations(active_models.keys(), 2):
        a_scores = active_models[a][:, i]
        b_scores = active_models[b][:, i]
        t, p = stats.ttest_rel(a_scores, b_scores)
        pairs.append((a, b))
        raw_ts.append(t)
        raw_ps.append(p)
        mean_diffs.append(a_scores.mean() - b_scores.mean())

    reject, p_corr, _, _ = multipletests(raw_ps, alpha=ALPHA, method=CORRECTION)

    print(f"    {'Comparison':<25} {'mean diff':>12} {'t':>8} {'p (raw)':>10} {'p (adj)':>10} {'sig':>6}")
    for (a, b), md, t, p_raw, p_adj in zip(pairs, mean_diffs, raw_ts, raw_ps, p_corr):
        comparison = f"{a} vs {b}"
        print(f"    {comparison:<25} {md:>+12.6f} {t:>8.3f} {p_raw:>10.6f} {p_adj:>10.6f} {sig_stars(p_adj):>6}")

print(f"\n{'='*100}")
print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
print("Mean diff is (first model − second model). Positive = first is better.")
print('='*100)

EOF
Output

====================================================================================================
REPEATED MEASURES ANOVA + PAIRED POST-HOC TESTS (correction: holm)
Models: ['MolTrans', 'DLM-DTI', 'MocFormer', '3DICE']
Replications per model: 5
====================================================================================================

[WARNING] Skipping models with placeholder NaN data: {'DLM-DTI', 'MolTrans'}
          Running with 2 models: ['MocFormer', '3DICE']


====================================================================================================
METRIC: ACC
====================================================================================================
  Model              Mean        Std
  MocFormer      0.797145   0.002109
  3DICE          0.809089   0.001671

  RM-ANOVA: F = 60.2696, p = 0.001484  **

  Post-hoc paired t-tests (holm-corrected):
    Comparison                   mean diff        t    p (raw)    p (adj)    sig
    MocFormer vs 3DICE           -0.011944   -7.763   0.001484   0.001484     **

====================================================================================================
METRIC: PPV
====================================================================================================
  Model              Mean        Std
  MocFormer      0.774049   0.003084
  3DICE          0.788900   0.006722

  RM-ANOVA: F = 24.4418, p = 0.007795  **

  Post-hoc paired t-tests (holm-corrected):
    Comparison                   mean diff        t    p (raw)    p (adj)    sig
    MocFormer vs 3DICE           -0.014851   -4.944   0.007795   0.007795     **

====================================================================================================
METRIC: TPR
====================================================================================================
  Model              Mean        Std
  MocFormer      0.846272   0.004507
  3DICE          0.855460   0.007867

  RM-ANOVA: F = 5.8650, p = 0.072623  ns
  → Omnibus test not significant. Skipping post-hoc.

====================================================================================================
METRIC: F1
====================================================================================================
  Model              Mean        Std
  MocFormer      0.808063   0.002679
  3DICE          0.815520   0.001535

  RM-ANOVA: F = 30.4717, p = 0.005258  **

  Post-hoc paired t-tests (holm-corrected):
    Comparison                   mean diff        t    p (raw)    p (adj)    sig
    MocFormer vs 3DICE           -0.007457   -5.520   0.005258   0.005258     **

====================================================================================================
METRIC: MCC
====================================================================================================
  Model              Mean        Std
  MocFormer      0.597576   0.002867
  3DICE          0.619880   0.002458

  RM-ANOVA: F = 95.7668, p = 0.000611  ***

  Post-hoc paired t-tests (holm-corrected):
    Comparison                   mean diff        t    p (raw)    p (adj)    sig
    MocFormer vs 3DICE           -0.022304   -9.786   0.000611   0.000611    ***

====================================================================================================
METRIC: AUROC
====================================================================================================
  Model              Mean        Std
  MocFormer      0.881571   0.001931
  3DICE          0.878720   0.004928

  RM-ANOVA: F = 1.0685, p = 0.359681  ns
  → Omnibus test not significant. Skipping post-hoc.

====================================================================================================
METRIC: AUPRC
====================================================================================================
  Model              Mean        Std
  MocFormer      0.877947   0.007078
  3DICE          0.867540   0.005105

  RM-ANOVA: F = 4.5577, p = 0.099659  ns
  → Omnibus test not significant. Skipping post-hoc.

====================================================================================================
Significance: * p<0.05, ** p<0.01, *** p<0.001
Mean diff is (first model − second model). Positive = first is better.
====================================================================================================
Done
The script is structured to plug in your MolTrans and DLM-DTI numbers directly. Just replace the [np.nan]*7 rows in the MOLTRANS and DLM_DTI arrays with your actual replicate scores (in the same column order: ACC, PPV, TPR, F1, MCC, AUROC, AUPRC), and re-run.

How the pipeline works once all 4 models are filled in:

Per metric, RM-ANOVA tests whether any of the 4 models differ from the others. F-statistic and p-value are reported.
If the omnibus is significant (p < 0.05), the script runs all 6 pairwise paired t-tests (4 choose 2 = 6 comparisons): MolTrans vs DLM-DTI, MolTrans vs MocFormer, MolTrans vs 3DICE, DLM-DTI vs MocFormer, DLM-DTI vs 3DICE, MocFormer vs 3DICE.
Holm correction is applied to control the family-wise error rate across the 6 comparisons. Holm is uniformly more powerful than Bonferroni at the same FWER, so I've defaulted to it — switch to 'bonferroni' if a reviewer specifically asks for it.
If omnibus is not significant, post-hoc tests are skipped — this is the standard "protected" ANOVA procedure, which preserves the type-I error rate.
A few practical notes for when you run with all 4 models:

The current 2-model output is essentially a sanity check — F = t² for paired t-test, so RM-ANOVA p-values exactly match the paired t-test p-values you saw earlier.
With 4 models and only 5 runs you have 12 residual degrees of freedom — adequate but not abundant. If a near-significant comparison is critical, increasing replications is the cleanest fix.
In your write-up, you can present this as a per-metric table with three columns: ANOVA p, then the row of 6 corrected pairwise p-values. Or annotate the existing performance tables with letter groupings (models sharing a letter are not significantly different).






