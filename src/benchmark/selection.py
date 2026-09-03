"""The two model selection rules.

The rule used in the paper picks, for each task, the algorithm with the highest
cross-validated AUC on the training set. It is fixed before the test set is
examined, so the test AUC we report is not the quantity that picked the model.

The alternative rule, picking the algorithm with the highest test AUC, is common
in practice and is computed here only so the two can be compared. It is never
used to report performance.
"""

import pandas as pd

from . import config as cfg


def select_on_cv(results):
    """Highest cross-validated AUC per task. This is the rule used in the paper."""
    return (results.dropna(subset=['CV AUC mean'])
            .sort_values('CV AUC mean', ascending=False)
            .groupby(['Outcome', 'Variant', 'Sex'], observed=True)
            .head(1)
            .sort_values(['Outcome', 'Variant', 'Sex'])
            .reset_index(drop=True))


def select_on_test(results):
    """Highest test AUC per task. Reported for comparison only."""
    return (results.dropna(subset=['AUC'])
            .sort_values('AUC', ascending=False)
            .groupby(['Outcome', 'Variant', 'Sex'], observed=True)
            .head(1)
            .sort_values(['Outcome', 'Variant', 'Sex'])
            .reset_index(drop=True))


def compare_rules(results):
    """What changes when the reported model is chosen on the test set instead.

    Two things are worth separating. The gain in reported AUC is what people
    usually worry about. The change in which model gets reported is the part that
    matters more here, because models that are indistinguishable on discrimination
    can differ a great deal in calibration.
    """
    results = cfg.add_task_column(results)
    rows = []
    for task in sorted(results.Task.unique()):
        g = results[results.Task == task].dropna(subset=['AUC'])
        cv = g.loc[g['CV AUC mean'].idxmax()]
        test = g.loc[g['AUC'].idxmax()]
        rows.append({
            'Task': task, 'Outcome': cv.Outcome, 'Variant': cv.Variant, 'Sex': cv.Sex,
            'Model selected on CV': cv.Model,
            'CV rule test AUC': round(cv['AUC'], 4),
            'CV rule AUC CI low': round(cv['AUC CI low'], 4),
            'CV rule AUC CI high': round(cv['AUC CI high'], 4),
            'CV rule calibration slope': round(cv['Calibration slope'], 4),
            'Model selected on test': test.Model,
            'Test rule test AUC': round(test['AUC'], 4),
            'Test rule AUC CI low': round(test['AUC CI low'], 4),
            'Test rule AUC CI high': round(test['AUC CI high'], 4),
            'Test rule calibration slope': round(test['Calibration slope'], 4),
            'AUC gain from test selection': round(test['AUC'] - cv['AUC'], 4),
            'Rules disagree': bool(cv.Model != test.Model),
        })
    return pd.DataFrame(rows)


def compare_with_logistic(results, selected):
    """Selected model against penalised logistic regression on the same tasks.

    The comparison that decides whether the additional complexity of these models
    is earning anything. If the difference is small, that is a finding worth
    reporting plainly rather than one to bury.
    """
    logit = (results[results['Model'] == 'Penalised logistic regression']
             .set_index(['Outcome', 'Variant', 'Sex'])[['CV AUC mean', 'AUC']])
    chosen = selected.set_index(['Outcome', 'Variant', 'Sex'])[['Model', 'CV AUC mean', 'AUC']]
    out = chosen.join(logit, rsuffix=' logistic')
    out['AUC gain'] = out['AUC'] - out['AUC logistic']
    return out.reset_index()
