"""The selection rules, and the published numbers they produce."""

from benchmark.selection import compare_rules, select_on_cv, select_on_test


def test_cv_rule_returns_one_model_per_task(results):
    selected = select_on_cv(results)
    assert len(selected) == 8
    assert selected.groupby(['Outcome', 'Variant', 'Sex']).size().eq(1).all()


def test_cv_rule_never_picks_a_lower_cross_validated_auc(results):
    selected = select_on_cv(results).set_index(['Outcome', 'Variant', 'Sex'])
    for key, group in results.groupby(['Outcome', 'Variant', 'Sex'], observed=True):
        assert selected.loc[key, 'CV AUC mean'] == group['CV AUC mean'].max()


def test_test_rule_never_picks_a_lower_test_auc(results):
    selected = select_on_test(results).set_index(['Outcome', 'Variant', 'Sex'])
    for key, group in results.groupby(['Outcome', 'Variant', 'Sex'], observed=True):
        assert selected.loc[key, 'AUC'] == group['AUC'].max()


def test_selecting_on_the_test_set_never_lowers_the_reported_auc(results):
    comparison = compare_rules(results)
    assert (comparison['AUC gain from test selection'] >= 0).all()


def test_published_selection_rule_numbers(results):
    # The paper reports a mean gain of 0.003 and disagreement in four of eight
    # tasks. These are the numbers a reader would check first.
    comparison = compare_rules(results)
    assert round(comparison['AUC gain from test selection'].mean(), 3) == 0.003
    assert int(comparison['Rules disagree'].sum()) == 4


def test_published_headline_numbers(results):
    # Seventy two models, and the AUC and calibration ranges quoted in the text.
    assert len(results) == 72
    assert round(results['AUC'].min(), 3) == 0.527
    assert round(results['AUC'].max(), 3) == 0.825
    assert round(results['Calibration slope'].min(), 2) == 0.02
    assert round(results['Calibration slope'].max(), 2) == 1.43

    in_band = results['Calibration slope'].between(0.80, 1.20).sum()
    assert in_band == 28


def test_penalised_regression_is_best_in_four_tasks(results):
    best = select_on_test(results)
    n = (best['Model'] == 'Penalised logistic regression').sum()
    assert n == 4
