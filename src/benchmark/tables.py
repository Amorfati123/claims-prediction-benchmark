"""Regenerates the tables in the paper from results/all_results.csv.

Table 2 is the search space and is written from the algorithm definitions rather
than from results, so the table and the code cannot drift apart.
"""

import numpy as np
import pandas as pd

from . import config as cfg
from .selection import compare_rules

TASK_ORDER = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
VARIANT_LABEL = {'PostIndex': 'Full', 'Baseline': 'Discharge only'}


def _matrix(results, value, decimals):
    """One algorithm per row, one task per column."""
    wide = (results.pivot_table(index='Model', columns='Task', values=value, observed=True)
            .reindex(cfg.ALGORITHM_ORDER)[TASK_ORDER]
            .round(decimals))
    wide.index = [cfg.display(m) for m in wide.index]
    return wide


def _round_half_up(value, decimals):
    """Python rounds halves to even, which turns 23.85 into 23.8.

    Percentages here land on exact halves often enough that it is worth being
    explicit, so the published tables and the regenerated ones agree.
    """
    factor = 10 ** decimals
    return np.floor(value * factor + 0.5) / factor


def table1_tasks(split_summary):
    """The eight benchmark tasks.

    Events per predictor is computed from the event count shown in the same row,
    so the two columns cannot disagree with each other.
    """
    rows = []
    for _, r in split_summary.iterrows():
        events = int(r['Events train'])
        prevalence = 100 * r['Events overall'] / r['N total']
        rows.append({
            'Task': cfg.task_id(r['Outcome'], r['Variant'], r['Sex']),
            'Outcome': r['Outcome'],
            'Predictor set': VARIANT_LABEL[r['Variant']],
            'Stratum': r['Sex'],
            'Predictors': int(r['Predictors']),
            'Train n': int(r['N train']),
            'Test n': int(r['N test']),
            'Train events': events,
            'Prevalence, %': _round_half_up(prevalence, 1),
            'EPP': _round_half_up(events / r['Predictors'], 2),
        })
    return pd.DataFrame(rows).sort_values('Task').reset_index(drop=True)


def table2_search_spaces():
    """The hyperparameter search space for each algorithm."""
    from .algorithms import ALGORITHMS
    rows = []
    for name, spec in ALGORITHMS.items():
        described = '; '.join(
            f'{p.replace("clf__", "")}: {len(v)} values' for p, v in spec['params'].items())
        combinations = int(np.prod([len(v) for v in spec['params'].values()]))
        rows.append({
            'Algorithm': cfg.display(name),
            'Tuned parameters': len(spec['params']),
            'Grid size': f'{combinations:,}',
            'Parameters searched': described,
        })
    return pd.DataFrame(rows)


def table3_auc(results):
    """Test AUC for every algorithm in every task."""
    results = cfg.add_task_column(results)
    wide = _matrix(results, 'AUC', 3)
    wide['Mean'] = wide[TASK_ORDER].mean(axis=1).round(3)
    ranks = results.assign(rank=results.groupby('Task')['AUC'].rank(ascending=False))
    mean_rank = ranks.groupby('Model', observed=True)['rank'].mean().reindex(cfg.ALGORITHM_ORDER)
    wide['Mean rank'] = [round(mean_rank[m], 2) for m in cfg.ALGORITHM_ORDER]
    return wide.reset_index().rename(columns={'index': 'Algorithm'})


def table4_calibration(results):
    """Calibration slope for every algorithm in every task."""
    results = cfg.add_task_column(results)
    wide = _matrix(results, 'Calibration slope', 2)
    deviation = (results.assign(dev=(results['Calibration slope'] - 1).abs())
                 .groupby('Model', observed=True)['dev'].mean()
                 .reindex(cfg.ALGORITHM_ORDER))
    wide['Mean deviation'] = [round(deviation[m], 2) for m in cfg.ALGORITHM_ORDER]
    return wide.reset_index().rename(columns={'index': 'Algorithm'})


def table5_summary(results):
    """Algorithm level summary across the eight tasks."""
    results = cfg.add_task_column(results)
    results = results.assign(
        rank=results.groupby('Task')['AUC'].rank(ascending=False),
        optimism=results['Train AUC'] - results['AUC'])

    logistic = (results[results.Model == 'Penalised logistic regression']
                .set_index('Task')['AUC'])
    results = results.assign(
        vs_logistic=[a - logistic[t] for a, t in zip(results.AUC, results.Task)])

    rows = []
    for model in cfg.ALGORITHM_ORDER:
        g = results[results.Model == model]
        rows.append({
            'Algorithm': cfg.display(model),
            'Mean AUC': round(g.AUC.mean(), 3),
            'Difference': ('reference' if model == 'Penalised logistic regression'
                           else f'{g.vs_logistic.mean():+.3f}'),
            'Best in task': int((g['rank'] == 1).sum()),
            'Mean slope (range)': (f'{g["Calibration slope"].mean():.2f} '
                                   f'({g["Calibration slope"].min():.2f} to '
                                   f'{g["Calibration slope"].max():.2f})'),
            'Mean Brier': round(g.Brier.mean(), 3),
            'Optimism': round(g.optimism.mean(), 3),
        })
    return (pd.DataFrame(rows).sort_values('Mean AUC', ascending=False)
            .reset_index(drop=True))


def table6_selection(results):
    """What changes when the reported model is selected on the test set."""
    comparison = compare_rules(results)
    short = {'Penalised logistic regression': 'PLR', 'Gradient boosting': 'GB',
             'Histogram gradient boosting': 'HGB', 'Random forest': 'RF',
             'Bagging': 'Bag', 'XGBoost': 'XGB', 'SGD classifier': 'SGD',
             'Neural network': 'NN', 'K-nearest neighbors': 'kNN'}

    def interval(row, prefix):
        return (f'{row[prefix + " test AUC"]:.3f} '
                f'({row[prefix + " AUC CI low"]:.3f} to {row[prefix + " AUC CI high"]:.3f})')

    rows = []
    for _, r in comparison.iterrows():
        rows.append({
            'Task': r['Task'],
            'Selected on CV': short[r['Model selected on CV']],
            'AUC (95% CI)': interval(r, 'CV rule'),
            'Slope': round(r['CV rule calibration slope'], 2),
            'Selected on test': short[r['Model selected on test']],
            'AUC (95% CI) ': interval(r, 'Test rule'),
            'Slope ': round(r['Test rule calibration slope'], 2),
            'AUC gain': round(r['AUC gain from test selection'], 3),
        })
    return pd.DataFrame(rows)


def write_all(results, split_summary, outdir=None):
    outdir = outdir or cfg.TABLES_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    tables = {
        'table1_tasks': table1_tasks(split_summary),
        'table2_search_spaces': table2_search_spaces(),
        'table3_auc': table3_auc(results),
        'table4_calibration': table4_calibration(results),
        'table5_summary': table5_summary(results),
        'table6_selection_rule': table6_selection(results),
    }
    for name, frame in tables.items():
        frame.to_csv(outdir / f'{name}.csv', index=False)
    return tables
