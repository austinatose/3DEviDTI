
import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from pathlib import Path

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
CORRECTION = 'holm'
EXPERIMENT_NAME = 'DrugBank'
PLOT_DIR = Path(__file__).resolve().parents[1] / 'stat_test_plots'

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

def slugify(text):
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', text.strip())
    return slug.strip('_') or 'experiment'

def coerce_model_arrays(raw_models):
    models = {}
    for model_name, values in raw_models.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Model '{model_name}' must be a 2D array: runs x metrics.")
        models[model_name] = arr
    return models

def normalize_experiment(raw_exp, fallback_name, fallback_metrics=None):
    if not isinstance(raw_exp, dict):
        raise ValueError(f"Experiment '{fallback_name}' must be a JSON object.")

    raw_models = raw_exp.get('models') or raw_exp.get('models_data') or raw_exp.get('results')
    if raw_models is None:
        raise ValueError(
            f"Experiment '{fallback_name}' must contain a 'models', 'models_data', or 'results' object."
        )

    exp_metrics = raw_exp.get('metrics', fallback_metrics or metrics)
    exp_name = raw_exp.get('name') or raw_exp.get('experiment_name') or fallback_name
    exp_reference = raw_exp.get('reference') or raw_exp.get('baseline_reference') or REFERENCE

    return {
        'name': str(exp_name),
        'metrics': [str(metric) for metric in exp_metrics],
        'models_data': coerce_model_arrays(raw_models),
        'reference': str(exp_reference),
    }

def load_experiments_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if 'experiments' not in payload:
        return [normalize_experiment(payload, payload.get('name', Path(json_path).stem))]

    experiments = payload['experiments']
    shared_metrics = payload.get('metrics')

    if isinstance(experiments, dict):
        return [
            normalize_experiment(exp_payload, exp_name, shared_metrics)
            for exp_name, exp_payload in experiments.items()
        ]

    if isinstance(experiments, list):
        return [
            normalize_experiment(exp_payload, f'experiment_{idx + 1}', shared_metrics)
            for idx, exp_payload in enumerate(experiments)
        ]

    raise ValueError("'experiments' must be either an object or a list.")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Repeated-measures ANOVA, paired t-tests, MCSim heatmaps, and CI forest plots.'
    )
    parser.add_argument(
        '--json',
        type=Path,
        help='Optional JSON file with experiment results. If omitted, built-in DrugBank arrays are used.',
    )
    parser.add_argument(
        '--experiment-name',
        help='Display/output name for a single experiment. For multi-experiment JSON, names in JSON are used.',
    )
    parser.add_argument(
        '--reference',
        help='Reference model to compare against baselines. Can also be set per experiment in JSON.',
    )
    parser.add_argument(
        '--plot-dir',
        type=Path,
        default=PLOT_DIR,
        help='Directory where plots will be written.',
    )
    return parser.parse_args()

def paired_ci(diff_values, confidence=0.95):
    """95% CI for the mean paired difference."""
    diff_values = np.asarray(diff_values, dtype=float)
    n = diff_values.shape[0]
    mean = diff_values.mean()
    sem = stats.sem(diff_values)
    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - tcrit * sem, mean + tcrit * sem

def all_pairwise_results(model_names, metric_idx):
    """Pairwise paired t-tests for one metric, Holm-corrected across all pairs."""
    pair_rows = []
    raw_ps = []

    for left, right in combinations(model_names, 2):
        left_scores = active_models[left][:, metric_idx]
        right_scores = active_models[right][:, metric_idx]
        diff_values = left_scores - right_scores
        t_stat, p_raw = stats.ttest_rel(left_scores, right_scores)
        mean_diff, ci_low, ci_high = paired_ci(diff_values)
        pair_rows.append({
            'left': left,
            'right': right,
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            't': t_stat,
            'p_raw': p_raw,
        })
        raw_ps.append(p_raw)

    _, p_adj, _, _ = multipletests(raw_ps, alpha=ALPHA, method=CORRECTION)
    for row, p in zip(pair_rows, p_adj):
        row['p_adj'] = p
        row['p'] = p
        row['stars'] = sig_stars(p)
    return pair_rows

def reference_pairwise_results(metric_idx):
    """REFERENCE-vs-baseline paired differences with Holm-corrected p-values and CIs."""
    rows = []
    raw_ps = []
    ref_scores = active_models[REFERENCE][:, metric_idx]

    for baseline_name, baseline_data in baselines.items():
        baseline_scores = baseline_data[:, metric_idx]
        diff_values = ref_scores - baseline_scores
        t_stat, p_raw = stats.ttest_rel(ref_scores, baseline_scores)
        mean_diff, ci_low, ci_high = paired_ci(diff_values)
        rows.append({
            'baseline': baseline_name,
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            't': t_stat,
            'p_raw': p_raw,
        })
        raw_ps.append(p_raw)

    _, p_adj, _, _ = multipletests(raw_ps, alpha=ALPHA, method=CORRECTION)
    for row, p in zip(rows, p_adj):
        row['p_adj'] = p
        row['p'] = p
        row['stars'] = sig_stars(p)
    return rows

def plot_mcsim_heatmap(metric_name, metric_idx, out_dir):
    """Main-paper heatmap: cell colour is row-model minus column-model mean."""
    model_names = sorted(
        active_models,
        key=lambda name: active_models[name][:, metric_idx].mean(),
        reverse=True,
    )
    n_models = len(model_names)
    mean_diff_matrix = np.zeros((n_models, n_models))
    star_matrix = np.full((n_models, n_models), '', dtype=object)

    for row_idx, row_name in enumerate(model_names):
        for col_idx, col_name in enumerate(model_names):
            mean_diff_matrix[row_idx, col_idx] = (
                active_models[row_name][:, metric_idx].mean()
                - active_models[col_name][:, metric_idx].mean()
            )

    for row in all_pairwise_results(model_names, metric_idx):
        left_idx = model_names.index(row['left'])
        right_idx = model_names.index(row['right'])
        star = '' if row['stars'] == 'ns' else row['stars']
        star_matrix[left_idx, right_idx] = star
        star_matrix[right_idx, left_idx] = star

    vmax = max(abs(mean_diff_matrix.min()), abs(mean_diff_matrix.max()))
    if vmax == 0:
        vmax = 1e-6

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(mean_diff_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_models), labels=model_names, rotation=35, ha='right')
    ax.set_yticks(range(n_models), labels=model_names)
    ax.set_title(f'{metric_name}: pairwise mean differences')
    ax.set_xlabel('Column model')
    ax.set_ylabel('Row model')

    for row_idx in range(n_models):
        for col_idx in range(n_models):
            if row_idx == col_idx:
                text = '0'
            else:
                text = f'{mean_diff_matrix[row_idx, col_idx]:+.4f}\n{star_matrix[row_idx, col_idx]}'
            ax.text(col_idx, row_idx, text, ha='center', va='center', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean difference (row - column)')
    fig.tight_layout()
    fig.savefig(out_dir / f'mcsim_heatmap_{metric_name}.png', dpi=300)
    plt.close(fig)

def plot_all_mcsim_heatmaps(out_dir):
    """Grid of all MCSim-style heatmaps, one panel per metric."""
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx]
        model_names = sorted(
            active_models,
            key=lambda name: active_models[name][:, metric_idx].mean(),
            reverse=True,
        )
        n_models = len(model_names)
        mean_diff_matrix = np.zeros((n_models, n_models))
        star_matrix = np.full((n_models, n_models), '', dtype=object)

        for row_idx, row_name in enumerate(model_names):
            for col_idx, col_name in enumerate(model_names):
                mean_diff_matrix[row_idx, col_idx] = (
                    active_models[row_name][:, metric_idx].mean()
                    - active_models[col_name][:, metric_idx].mean()
                )

        for row in all_pairwise_results(model_names, metric_idx):
            left_idx = model_names.index(row['left'])
            right_idx = model_names.index(row['right'])
            star = '' if row['stars'] == 'ns' else row['stars']
            star_matrix[left_idx, right_idx] = star
            star_matrix[right_idx, left_idx] = star

        vmax = max(abs(mean_diff_matrix.min()), abs(mean_diff_matrix.max()))
        if vmax == 0:
            vmax = 1e-6
        im = ax.imshow(mean_diff_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(metric_name)
        ax.set_xticks(range(n_models), labels=model_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_models), labels=model_names, fontsize=8)

        for row_idx in range(n_models):
            for col_idx in range(n_models):
                text = '0' if row_idx == col_idx else f'{mean_diff_matrix[row_idx, col_idx]:+.3f}\n{star_matrix[row_idx, col_idx]}'
                ax.text(col_idx, row_idx, text, ha='center', va='center', fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_metrics:]:
        ax.axis('off')

    fig.suptitle('Pairwise model comparisons: mean difference (row - column)', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'mcsim_heatmaps_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_reference_forest(metric_name, metric_idx, out_dir):
    """Forest plot for REFERENCE vs every baseline."""
    rows = sorted(reference_pairwise_results(metric_idx), key=lambda row: row['mean_diff'])
    labels = [f"{REFERENCE} vs {row['baseline']}" for row in rows]
    means = np.array([row['mean_diff'] for row in rows])
    ci_low = np.array([row['ci_low'] for row in rows])
    ci_high = np.array([row['ci_high'] for row in rows])
    stars = [row['stars'] for row in rows]
    y_pos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 0.65 * len(rows) + 1.8))
    xerr = np.vstack([means - ci_low, ci_high - means])
    colors = ['#1b9e77' if low > 0 or high < 0 else '#757575' for low, high in zip(ci_low, ci_high)]
    ax.errorbar(means, y_pos, xerr=xerr, fmt='none', ecolor='#404040', capsize=4, lw=1.4)
    ax.scatter(means, y_pos, c=colors, s=55, zorder=3)
    ax.axvline(0, color='black', lw=1, linestyle='--')
    ax.set_yticks(y_pos, labels=labels)
    ax.set_xlabel(f'Mean difference in {metric_name} ({REFERENCE} - baseline)')
    ax.set_title(f'{metric_name}: {REFERENCE} vs baselines, 95% CI')
    ax.grid(axis='x', alpha=0.25)

    x_span = ci_high.max() - ci_low.min()
    label_x = ci_high.max() + 0.05 * x_span if x_span > 0 else ci_high.max() + 0.001
    for y, star in zip(y_pos, stars):
        ax.text(label_x, y, star, va='center', fontsize=10)
    ax.set_xlim(ci_low.min() - 0.15 * x_span, label_x + 0.15 * x_span)

    fig.tight_layout()
    fig.savefig(out_dir / f'forest_ci_{metric_name}.png', dpi=300)
    plt.close(fig)

def plot_all_reference_forests(out_dir):
    """Grid of forest plots for all metrics."""
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.0 * n_cols, 2.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx]
        rows = sorted(reference_pairwise_results(metric_idx), key=lambda row: row['mean_diff'])
        labels = [row['baseline'] for row in rows]
        means = np.array([row['mean_diff'] for row in rows])
        ci_low = np.array([row['ci_low'] for row in rows])
        ci_high = np.array([row['ci_high'] for row in rows])
        stars = [row['stars'] for row in rows]
        y_pos = np.arange(len(rows))
        xerr = np.vstack([means - ci_low, ci_high - means])
        colors = ['#1b9e77' if low > 0 or high < 0 else '#757575' for low, high in zip(ci_low, ci_high)]

        ax.errorbar(means, y_pos, xerr=xerr, fmt='none', ecolor='#404040', capsize=3, lw=1.2)
        ax.scatter(means, y_pos, c=colors, s=42, zorder=3)
        ax.axvline(0, color='black', lw=1, linestyle='--')
        ax.set_yticks(y_pos, labels=labels)
        ax.set_title(metric_name)
        ax.grid(axis='x', alpha=0.25)
        ax.set_xlabel(f'{REFERENCE} - baseline')

        x_span = ci_high.max() - ci_low.min()
        label_x = ci_high.max() + 0.05 * x_span if x_span > 0 else ci_high.max() + 0.001
        for y, star in zip(y_pos, stars):
            ax.text(label_x, y, star, va='center', fontsize=9)
        ax.set_xlim(ci_low.min() - 0.15 * x_span, label_x + 0.15 * x_span)

    for ax in axes[n_metrics:]:
        ax.axis('off')

    fig.suptitle(f'{REFERENCE} vs baselines: mean difference with 95% CI', y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'forest_ci_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def make_plots(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # for metric_idx, metric_name in enumerate(metrics):
    #     plot_mcsim_heatmap(metric_name, metric_idx, out_dir)
    #     plot_reference_forest(metric_name, metric_idx, out_dir)
    plot_all_mcsim_heatmaps(out_dir)
    plot_all_reference_forests(out_dir)
    print(f"\nSaved MCSim heatmaps and forest CI plots to: {out_dir.resolve()}")

# =====================================================================
# Per-metric analysis
# =====================================================================

def validate_experiment_shapes():
    if REFERENCE not in models_data:
        raise ValueError(f"Reference model '{REFERENCE}' is not present in models_data.")

    if len(models_data) < 2:
        raise ValueError("At least two models are required.")

    expected_n_metrics = len(metrics)
    run_counts = set()
    for model_name, data in models_data.items():
        if data.shape[1] != expected_n_metrics:
            raise ValueError(
                f"Model '{model_name}' has {data.shape[1]} metrics, "
                f"but metrics lists {expected_n_metrics} names."
            )
        run_counts.add(data.shape[0])

    if len(run_counts) != 1:
        raise ValueError(f"All models must have the same number of runs. Got run counts: {sorted(run_counts)}")

    return run_counts.pop()

def run_experiment(experiment, base_plot_dir):
    global metrics, models_data, REFERENCE, active_models, baselines, ref_data

    metrics = experiment['metrics']
    models_data = experiment['models_data']
    REFERENCE = experiment['reference']
    experiment_name = experiment['name']
    n_runs = validate_experiment_shapes()

    active_models = {k: v for k, v in models_data.items() if has_data(v)}
    baselines = {k: v for k, v in active_models.items() if k != REFERENCE}
    ref_data = active_models[REFERENCE]

    out_dir = base_plot_dir / slugify(experiment_name)

    print("="*100)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"REPEATED MEASURES ANOVA + POST-HOC: {REFERENCE} vs each baseline ({CORRECTION}-corrected)")
    print(f"Models: {list(active_models.keys())}")
    print(f"Replications per model: {n_runs}")
    print("="*100)

    for i, m in enumerate(metrics):
        print(f"\n{'='*100}")
        print(f"METRIC: {m}")
        print('='*100)

        # Mean +/- std for all models
        print(f"  {'Model':<12} {'Mean':>10} {'Std':>10}")
        for name, data in active_models.items():
            col = data[:, i]
            marker = ' ◄' if name == REFERENCE else ''
            print(f"  {name:<12} {col.mean():>10.6f} {col.std(ddof=1):>10.6f}{marker}")

        # --- RM-ANOVA (omnibus across all models) ---
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
            print("  -> Omnibus test not significant. Skipping post-hoc.")
            continue

        # --- Post-hoc: reference model vs each baseline only ---
        print(f"\n  Post-hoc: {REFERENCE} vs each baseline ({CORRECTION}-corrected across {len(baselines)} comparisons):")

        pairs, raw_ps, raw_ts, mean_diffs = [], [], [], []
        for baseline_name, baseline_data in baselines.items():
            ref_scores = ref_data[:, i]
            baseline_scores = baseline_data[:, i]
            t, p = stats.ttest_rel(ref_scores, baseline_scores)
            pairs.append(baseline_name)
            raw_ts.append(t)
            raw_ps.append(p)
            mean_diffs.append(ref_scores.mean() - baseline_scores.mean())

        _, p_corr, _, _ = multipletests(raw_ps, alpha=ALPHA, method=CORRECTION)

        print(f"    {'Comparison':<30} {f'{REFERENCE}-baseline':>15} {'t':>8} {'p (raw)':>10} {'p (adj)':>10} {'result':>16}")
        for baseline_name, md, t, p_raw, p_adj in zip(pairs, mean_diffs, raw_ts, raw_ps, p_corr):
            comparison = f"{REFERENCE} vs {baseline_name}"
            direc = direction(md)
            label = f"{sig_stars(p_adj)} ({direc})" if p_adj < ALPHA else 'ns'
            print(f"    {comparison:<30} {md:>+15.6f} {t:>8.3f} {p_raw:>10.6f} {p_adj:>10.6f} {label:>16}")

    print(f"\n{'='*100}")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"{REFERENCE}-baseline > 0 means {REFERENCE} is better than the baseline.")
    print('='*100)

    make_plots(out_dir)

def main():
    args = parse_args()

    if args.json:
        experiments = load_experiments_from_json(args.json)
        if args.experiment_name:
            if len(experiments) != 1:
                raise ValueError("--experiment-name can only override a JSON file with one experiment.")
            experiments[0]['name'] = args.experiment_name
        if args.reference:
            for exp in experiments:
                exp['reference'] = args.reference
    else:
        experiments = [{
            'name': args.experiment_name or EXPERIMENT_NAME,
            'metrics': metrics,
            'models_data': models_data,
            'reference': args.reference or REFERENCE,
        }]

    for idx, experiment in enumerate(experiments):
        if idx > 0:
            print("\n")
        run_experiment(experiment, args.plot_dir)

if __name__ == '__main__':
    main()
