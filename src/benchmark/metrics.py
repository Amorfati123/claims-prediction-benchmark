"""How performance is measured.

Discrimination is the area under the ROC curve, which does not depend on where
the probability is cut. Everything else does depend on that cut, and two cut
points are reported: the conventional 0.5, and one chosen on the training set by
maximising Youden's index on cross-validated predictions.

Calibration is reported as the Brier score together with the slope and intercept
of the observed outcome regressed on the predicted log odds. A model can
discriminate well and still be badly calibrated, in which case its probabilities
cannot be read as risks. That gap is the subject of the paper, so the two are
always computed together and never reported apart.
"""

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .config import CALIBRATION_BINS, N_BOOTSTRAP, RANDOM_STATE


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(num, den):
        # A model that never predicts one class leaves a denominator at zero.
        # Reporting that as missing is honest, reporting it as zero is not.
        return float(num) / den if den else np.nan

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': ratio(tp, tp + fn),
        'Specificity': ratio(tn, tn + fp),
        'PPV': ratio(tp, tp + fp),
        'NPV': ratio(tn, tn + fn),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'F1 macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


def calibration_metrics(y_true, y_score):
    """Brier score plus the slope and intercept of a logistic recalibration."""
    out = {'Brier': brier_score_loss(y_true, y_score)}

    # Predictions are nudged off 0 and 1 so the log odds stay finite.
    p = np.clip(y_score, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))

    try:
        # Regressing the outcome on the predicted log odds. A slope of one means
        # the spread of predicted risk matches the spread of observed risk, below
        # one means the predictions are too extreme.
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        model.fit(logit.reshape(-1, 1), y_true)
        out['Calibration slope'] = float(model.coef_[0][0])
    except Exception:
        out['Calibration slope'] = np.nan

    try:
        # The intercept is taken with the slope held at one, so it isolates a
        # systematic offset in risk from a difference in spread. It is the shift
        # that makes the mean predicted risk equal the observed rate, solved for
        # directly.
        def offset_gap(a):
            return np.mean(1.0 / (1.0 + np.exp(-(a + logit)))) - y_true.mean()

        out['Calibration intercept'] = float(brentq(offset_gap, -20, 20))
    except Exception:
        out['Calibration intercept'] = np.nan

    return out


def youden_threshold(y_true, y_score):
    """Probability cut point maximising sensitivity plus specificity minus one."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def bootstrap_intervals(y_true, y_score, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Percentile confidence intervals for the headline metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected = {'AUC': [], 'Sensitivity': [], 'Specificity': [], 'Brier': []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        # A resample that happens to contain one class only cannot produce these
        # metrics, so it is discarded rather than counted.
        if len(np.unique(yt)) < 2:
            continue
        yp = (ys >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        collected['AUC'].append(roc_auc_score(yt, ys))
        collected['Sensitivity'].append(tp / (tp + fn) if (tp + fn) else np.nan)
        collected['Specificity'].append(tn / (tn + fp) if (tn + fp) else np.nan)
        collected['Brier'].append(brier_score_loss(yt, ys))

    out = {}
    for metric, values in collected.items():
        values = np.array([v for v in values if not np.isnan(v)])
        if len(values):
            lo, hi = np.percentile(values, [2.5, 97.5])
            out[f'{metric} CI low'] = float(lo)
            out[f'{metric} CI high'] = float(hi)
        else:
            out[f'{metric} CI low'] = np.nan
            out[f'{metric} CI high'] = np.nan
    return out


def calibration_curve_points(y_true, y_score, n_bins=CALIBRATION_BINS):
    """Observed against predicted risk in equal count bins, for the calibration plots."""
    order = np.argsort(y_score)
    bins = np.array_split(order, n_bins)
    return [{'predicted': float(y_score[b].mean()), 'observed': float(y_true[b].mean()),
             'n': len(b)} for b in bins if len(b)]
