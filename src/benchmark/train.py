"""Tuning, fitting and scoring.

The routine below tunes an algorithm, refits it on the whole training set,
derives the decision threshold from cross-validated training predictions, and
scores the tuned model once on the test set.

The threshold is derived from out-of-fold predictions rather than from the fitted
model's own training predictions, because a model scoring the data it was fitted
on is overconfident and would place the cut point in the wrong spot.

Scaling happens inside the pipeline, so the scaler is refitted within each cross
validation fold and the fold being scored never contributes to its own
standardisation.
"""

import json
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .algorithms import ALGORITHMS
from .metrics import (bootstrap_intervals, calibration_curve_points, calibration_metrics,
                      classification_metrics, youden_threshold)

EMPTY_STATE = {'results': [], 'best_params': [], 'calibration_curve': []}


def checkpoint_path(key):
    return cfg.CHECKPOINT_DIR / f'stepBC_{key}.json'


def model_path(key, slug):
    return cfg.MODEL_DIR / f'stepBC_{key}_{slug}.joblib'


def load_checkpoint(key, resume=True):
    path = checkpoint_path(key)
    if resume and path.exists():
        with open(path) as f:
            return json.load(f)
    return {k: list(v) for k, v in EMPTY_STATE.items()}


def save_checkpoint(key, state):
    with open(checkpoint_path(key), 'w') as f:
        json.dump(state, f, indent=2)


def fit_one(key, meta, name, spec, data, save_model=True):
    d = data[key]
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clone(spec['estimator']))])

    search = RandomizedSearchCV(
        pipeline, spec['params'], n_iter=cfg.N_SEARCH_ITER, cv=cv,
        scoring={cfg.TUNING_METRIC: cfg.TUNING_METRIC, 'roc_auc': 'roc_auc'},
        refit=cfg.TUNING_METRIC, n_jobs=cfg.N_JOBS,
        random_state=cfg.RANDOM_STATE, error_score=np.nan)
    search.fit(d['X_train'], d['y_train'])

    best = search.best_estimator_
    if save_model:
        cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, model_path(key, spec['slug']), compress=3)

    # Out-of-fold training predictions, used only to choose the threshold.
    oof = cross_val_predict(clone(best), d['X_train'], d['y_train'],
                            cv=cv, method='predict_proba', n_jobs=cfg.N_JOBS)[:, 1]
    threshold = youden_threshold(d['y_train'], oof)

    test_score = best.predict_proba(d['X_test'])[:, 1]
    y_test = d['y_test']

    row = {
        'Outcome': meta['outcome'], 'Variant': meta['variant'], 'Sex': meta['sex'],
        'Model': name, 'Predictors': meta['n_features'],
        'N train': meta['n_train'], 'N test': meta['n_test'],
        'Prevalence test': meta['prevalence_test'],
        'AUC': roc_auc_score(y_test, test_score),
        'CV AUC mean': float(search.cv_results_['mean_test_roc_auc'][search.best_index_]),
        'CV AUC sd': float(search.cv_results_['std_test_roc_auc'][search.best_index_]),
        'CV F1 macro': float(
            search.cv_results_[f'mean_test_{cfg.TUNING_METRIC}'][search.best_index_]),
        'Train AUC': roc_auc_score(d['y_train'], best.predict_proba(d['X_train'])[:, 1]),
        'Threshold': threshold,
    }
    row.update(calibration_metrics(y_test, test_score))

    # Performance at the default cut point and at the one chosen on the training set.
    for label, cut in [('', 0.5), (' at threshold', threshold)]:
        metrics = classification_metrics(y_test, (test_score >= cut).astype(int))
        for metric, value in metrics.items():
            row[f'{metric}{label}'] = value

    row.update(bootstrap_intervals(y_test, test_score, threshold))

    params = [{
        'Outcome': meta['outcome'], 'Variant': meta['variant'], 'Sex': meta['sex'],
        'Model': name, 'Parameter': p.replace('clf__', ''), 'Value': str(v),
    } for p, v in sorted(search.best_params_.items())]

    curve = [{'Outcome': meta['outcome'], 'Variant': meta['variant'], 'Sex': meta['sex'],
              'Model': name, **point}
             for point in calibration_curve_points(y_test, test_score)]

    return row, params, curve


def run(configurations, data, resume=True, algorithms=None, save_model=True):
    """Fit every algorithm on every configuration.

    Results are written after each algorithm, so an interrupted run loses at most
    one model and picks up where it stopped when restarted.
    """
    algorithms = algorithms or ALGORITHMS
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for position, (key, meta) in enumerate(configurations.items(), start=1):
        state = load_checkpoint(key, resume=resume)
        done = {r['Model'] for r in state['results']}

        header = (f'[{position}/{len(configurations)}] {meta["outcome"]}, {meta["variant"]}, '
                  f'{meta["sex"]}  ({meta["n_features"]} predictors, '
                  f'{meta["n_train"]:,} training patients)')
        print(header)
        print('-' * len(header))

        for name, spec in algorithms.items():
            if name in done:
                print(f'  {name:32s} already done, skipped')
                continue

            started = time.time()
            try:
                row, params, curve = fit_one(key, meta, name, spec, data, save_model=save_model)
                row['Fit time (s)'] = round(time.time() - started, 1)
                state['results'].append(row)
                state['best_params'].extend(params)
                state['calibration_curve'].extend(curve)
                print(f'  {name:32s} AUC {row["AUC"]:.3f}   CV AUC {row["CV AUC mean"]:.3f}   '
                      f'slope {row["Calibration slope"]:.2f}   '
                      f'Brier {row["Brier"]:.3f}   [{row["Fit time (s)"]:.0f} s]')
            except Exception as error:
                # Recorded rather than raised, so one failure does not end an
                # overnight run and the gap shows up in the verification below.
                print(f'  {name:32s} FAILED after {time.time() - started:.0f} s: {error}')
                state['results'].append({
                    'Outcome': meta['outcome'], 'Variant': meta['variant'], 'Sex': meta['sex'],
                    'Model': name, 'Predictors': meta['n_features'],
                    'N train': meta['n_train'], 'N test': meta['n_test'],
                    'Prevalence test': meta['prevalence_test'],
                    'Fit time (s)': round(time.time() - started, 1),
                })

            save_checkpoint(key, state)
        print()


def collect(configurations):
    """Read the checkpoints back rather than reusing what is left in memory."""
    import pandas as pd

    results_rows, params_rows, curve_rows = [], [], []
    for key in configurations:
        state = load_checkpoint(key)
        results_rows.extend(state['results'])
        params_rows.extend(state['best_params'])
        curve_rows.extend(state['calibration_curve'])

    results = pd.DataFrame(results_rows)
    results['Outcome'] = pd.Categorical(results['Outcome'], list(cfg.OUTCOMES), ordered=True)
    results['Variant'] = pd.Categorical(results['Variant'], list(cfg.VARIANTS), ordered=True)
    results['Sex'] = pd.Categorical(results['Sex'], ['Women', 'Men'], ordered=True)
    results['Model'] = pd.Categorical(results['Model'], list(ALGORITHMS), ordered=True)
    results = results.sort_values(['Outcome', 'Variant', 'Sex', 'Model']).reset_index(drop=True)

    numeric = results.select_dtypes(include=[float]).columns
    results[numeric] = results[numeric].round(4)

    return results, pd.DataFrame(params_rows), pd.DataFrame(curve_rows)


def check_results(results, configurations, n_algorithms):
    """Six checks. A silent problem here would be carried into every number reported."""
    report = {}
    expected = len(configurations) * n_algorithms
    report['one result per configuration and algorithm'] = len(results) == expected
    report['every model produced a result'] = bool(results['AUC'].notna().all())
    report['metrics lie in a sensible range'] = bool(
        results['AUC'].between(0, 1).all() and results['Brier'].between(0, 1).all())

    sizes_ok = True
    for _, r in results.iterrows():
        meta = configurations[f'{r["Outcome"]}_{r["Variant"]}_{r["Sex"]}']
        if (r['N train'] != meta['n_train'] or r['N test'] != meta['n_test']
                or r['Predictors'] != meta['n_features']):
            sizes_ok = False
    report['sample sizes match the prepared configurations'] = sizes_ok

    collapsed = results[(results['Sensitivity at threshold'] == 0)
                        | (results['Specificity at threshold'] == 0)]
    report['no model collapsed onto a single predicted class'] = len(collapsed) == 0

    inside = ((results['AUC CI low'] <= results['AUC'])
              & (results['AUC'] <= results['AUC CI high']))
    report['AUC confidence interval contains the point estimate'] = bool(inside.all())
    return report
