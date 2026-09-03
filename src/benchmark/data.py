"""Builds the modelling matrices.

For each combination of outcome, predictor variant and sex this selects the
predictors, removes anything that would leak the answer, and splits the patients
into a training and a test set. Every algorithm then trains on exactly the same
rows and columns, so differences in performance come from the algorithms rather
than from the data they happened to see.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg


def load_domain_map():
    with open(cfg.DOMAIN_MAP_PATH) as f:
        return json.load(f)


def load_cohort(women_path, men_path):
    return {'Women': pd.read_csv(women_path), 'Men': pd.read_csv(men_path)}


def candidate_predictors(domain_map):
    """Every column the domain map assigns to a predictor domain."""
    return [c for c, dom in domain_map.items() if dom in cfg.PREDICTOR_DOMAINS]


def adherence_derived(domain_map):
    """Columns derived from adherence, excluded from every predictor set.

    It is tempting to remove only the outcome being modelled at the time, but
    the three outcomes are close relatives of each other and the continuous
    measures reconstruct them exactly, so any of them left in the matrix would
    let a model read the answer rather than predict it.
    """
    return sorted(c for c, dom in domain_map.items() if dom == 'outcome')


def constant_columns(frames, candidates):
    """Columns holding one value for everyone in a file.

    They carry no information within that stratum. They arise here because some
    conditions are specific to one sex, so the flag is zero for every patient in
    the other file.
    """
    return {sex: [c for c in candidates if d[c].nunique(dropna=False) <= 1]
            for sex, d in frames.items()}


def predictors_for(candidates, constants, sex, variant):
    drop = set(constants[sex]) | set(cfg.VARIANTS[variant])
    return [c for c in candidates if c not in drop]


def build_matrices(frames, domain_map, matrix_dir=None):
    """Write one compressed matrix per configuration and return the manifest.

    The patient identifiers are kept alongside the matrices. They are never used
    as a predictor, but keeping them means a patient can be traced back from a
    prediction if anything needs checking later.
    """
    matrix_dir = matrix_dir or cfg.MATRIX_DIR
    matrix_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_predictors(domain_map)
    constants = constant_columns(frames, candidates)

    configurations, split_rows = {}, []

    for outcome_name, outcome_col in cfg.OUTCOMES.items():
        for variant in cfg.VARIANTS:
            if variant == 'Baseline' and outcome_name not in cfg.BASELINE_OUTCOMES:
                continue
            for sex, d in frames.items():
                features = predictors_for(candidates, constants, sex, variant)
                X = d[features].to_numpy(dtype=float)
                y = d[outcome_col].to_numpy(dtype=int)
                ids = d['ID'].to_numpy()

                X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                    X, y, ids, test_size=cfg.TEST_SIZE,
                    random_state=cfg.RANDOM_STATE, stratify=y)

                key = f'{outcome_name}_{variant}_{sex}'
                path = matrix_dir / f'{key}.npz'
                np.savez_compressed(
                    path,
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    id_train=id_train, id_test=id_test,
                    features=np.array(features, dtype=object))

                configurations[key] = {
                    'outcome': outcome_name, 'outcome_column': outcome_col,
                    'variant': variant, 'sex': sex, 'file': str(path),
                    'n_features': len(features),
                    'n_total': len(y), 'n_train': len(y_train), 'n_test': len(y_test),
                    'prevalence_overall': float(y.mean()),
                    'prevalence_train': float(y_train.mean()),
                    'prevalence_test': float(y_test.mean()),
                }
                split_rows.append({
                    'Outcome': outcome_name, 'Variant': variant, 'Sex': sex,
                    'Predictors': len(features), 'N total': len(y),
                    'N train': len(y_train), 'N test': len(y_test),
                    'Prev overall': round(float(y.mean()), 4),
                    'Prev train': round(float(y_train.mean()), 4),
                    'Prev test': round(float(y_test.mean()), 4),
                    # Exact counts as well as proportions. A prevalence stored to
                    # four decimals is ambiguous at an exact half, which showed up
                    # as a one decimal disagreement in the published table.
                    'Events overall': int(y.sum()),
                    'Events train': int(y_train.sum()),
                })

    return configurations, pd.DataFrame(split_rows), constants


def load_matrices(configurations):
    """Read the matrices back from disk.

    It costs a moment and it guarantees that what gets trained on is what was
    written out and checked.
    """
    data = {}
    for key, meta in configurations.items():
        a = np.load(meta['file'], allow_pickle=True)
        data[key] = {
            'X_train': a['X_train'], 'X_test': a['X_test'],
            'y_train': a['y_train'], 'y_test': a['y_test'],
            'features': list(a['features']),
        }
        assert data[key]['X_train'].shape[1] == meta['n_features']
        assert data[key]['X_train'].shape[0] == meta['n_train']
    return data


def check_matrices(configurations, split_summary, excluded):
    """Four checks that a problem here would silently invalidate everything after."""
    report = {}

    shared_any = False
    for meta in configurations.values():
        a = np.load(meta['file'], allow_pickle=True)
        if set(a['id_train']) & set(a['id_test']):
            shared_any = True
    report['no patient in both train and test'] = not shared_any

    leaked_any = False
    for meta in configurations.values():
        a = np.load(meta['file'], allow_pickle=True)
        if set(a['features']) & set(excluded):
            leaked_any = True
    report['no adherence derived column in any predictor set'] = not leaked_any

    baseline_leak = False
    for meta in configurations.values():
        if meta['variant'] != 'Baseline':
            continue
        a = np.load(meta['file'], allow_pickle=True)
        if set(a['features']) & set(cfg.POST_INDEX_PREDICTORS):
            baseline_leak = True
    report['baseline variant excludes post-discharge measures'] = not baseline_leak

    worst = (split_summary['Prev train'] - split_summary['Prev test']).abs().max()
    report['stratification held within one percentage point'] = bool(worst < 0.01)

    return report
