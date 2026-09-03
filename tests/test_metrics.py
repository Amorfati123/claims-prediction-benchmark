"""Metric behaviour on cases where the right answer is known in advance."""

import numpy as np
import pytest

from benchmark.metrics import (calibration_metrics, classification_metrics,
                               youden_threshold)


def test_perfectly_calibrated_predictions_give_a_slope_near_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 40000)
    y = rng.binomial(1, p)
    out = calibration_metrics(y, p)
    assert out['Calibration slope'] == pytest.approx(1.0, abs=0.08)
    assert out['Calibration intercept'] == pytest.approx(0.0, abs=0.08)


def test_over_extreme_predictions_give_a_slope_below_one():
    # Pushing predictions towards 0 and 1 is the failure mode we report for
    # XGBoost, and it should show up as a slope well under one.
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 40000)
    y = rng.binomial(1, p)
    logit = np.log(p / (1 - p))
    sharpened = 1 / (1 + np.exp(-2.5 * logit))
    assert calibration_metrics(y, sharpened)['Calibration slope'] < 0.6


def test_a_model_that_predicts_one_class_reports_missing_rather_than_zero():
    y = np.array([0, 0, 1, 1])
    out = classification_metrics(y, np.zeros(4, dtype=int))
    assert np.isnan(out['PPV'])
    assert out['Sensitivity'] == 0.0
    assert out['TP'] == 0 and out['TN'] == 2


def test_youden_threshold_separates_a_clean_split():
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])
    assert 0.2 < youden_threshold(y, score) <= 0.8


def test_calibration_handles_predictions_at_zero_and_one():
    # Untransformed these would send the log odds to infinity.
    y = np.array([0, 1] * 500)
    score = np.array([0.0, 1.0] * 500)
    out = calibration_metrics(y, score)
    assert np.isfinite(out['Brier'])
