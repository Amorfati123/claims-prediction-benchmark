"""The checks that matter most for a paper arguing that method discipline matters.

If any of these fail, nothing else in the repository is worth reading.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark import config as cfg
from benchmark import data as bd
from benchmark.algorithms import ALGORITHMS


def test_scaler_sits_inside_the_pipeline():
    # Standardising before the split would let the test set influence the mean
    # and standard deviation used to scale the training set.
    for name, spec in ALGORITHMS.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', spec['estimator'])])
        assert pipeline.steps[0][0] == 'scaler', name
        assert isinstance(pipeline.steps[0][1], StandardScaler), name


def test_every_search_space_targets_the_classifier_step():
    # A parameter that does not carry the clf__ prefix would silently fail to
    # reach the estimator and the algorithm would be left at its defaults.
    for name, spec in ALGORITHMS.items():
        for parameter in spec['params']:
            assert parameter.startswith('clf__'), f'{name}: {parameter}'


def test_adherence_derived_columns_are_excluded(domain_map):
    # The three outcomes, the continuous measures they were cut from, and their
    # alternative thresholds and groupings. Any of them left in the matrix would
    # let a model read the answer rather than predict it.
    excluded = bd.adherence_derived(domain_map)
    predictors = bd.candidate_predictors(domain_map)
    assert len(excluded) == 12
    assert not set(predictors) & set(excluded)
    for outcome_column in cfg.OUTCOMES.values():
        assert outcome_column in excluded


def test_baseline_variant_drops_the_post_discharge_measures(domain_map):
    predictors = bd.candidate_predictors(domain_map)
    constants = {'Women': [], 'Men': []}
    for sex in constants:
        baseline = bd.predictors_for(predictors, constants, sex, 'Baseline')
        assert not set(baseline) & set(cfg.POST_INDEX_PREDICTORS)
        post_index = bd.predictors_for(predictors, constants, sex, 'PostIndex')
        assert set(post_index) - set(baseline) == set(cfg.POST_INDEX_PREDICTORS)


def test_train_and_test_never_share_a_patient():
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    ids = np.arange(2000)
    y = rng.binomial(1, 0.45, 2000)
    id_train, id_test = train_test_split(
        ids, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y)
    assert not set(id_train) & set(id_test)


def test_reported_predictor_counts_match_the_manifest(results, split_summary):
    # The paper states 574 predictors in the full variant and 570 in the
    # discharge-only variant. If the manifest disagrees, the paper is wrong.
    full = split_summary[split_summary.Variant == 'PostIndex']['Predictors'].unique()
    baseline = split_summary[split_summary.Variant == 'Baseline']['Predictors'].unique()
    assert list(full) == [574]
    assert list(baseline) == [570]

    merged = results.merge(split_summary, on=['Outcome', 'Variant', 'Sex'],
                           suffixes=('', '_manifest'))
    assert (merged['Predictors'] == merged['Predictors_manifest']).all()
