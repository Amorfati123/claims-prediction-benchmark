"""The nine algorithms and their search spaces.

Eight are machine learning methods of the kind usually applied to this sort of
problem. The ninth is penalised logistic regression, and it is there
deliberately. The question any reader will ask of a machine learning prediction
model is whether it does better than standard regression, and the only honest
way to answer that is to fit the regression on the same data, tune it with the
same budget, and put it in the same table. Elastic net penalisation lets it cope
with several hundred correlated predictors, which plain logistic regression
could not.

The search spaces are written as explicit lists so the sampled combinations are
reproducible.
"""

from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def value_range(start, stop, step, decimals=None):
    """Inclusive list of parameter values from start to stop."""
    n = int((stop - start) / step + 1e-6) + 1
    values = [start + i * step for i in range(n)]
    if decimals is not None:
        values = [round(v, decimals) for v in values]
    return values


N_ESTIMATORS = value_range(10, 200, 10)
FOREST_DEPTH = value_range(5, 500, 10)
BOOSTED_DEPTH = value_range(5, 50, 5)
MAX_FEATURES = value_range(0.02, 1.0, 0.02, 4)
SUBSAMPLE = value_range(0.1, 1.0, 0.1, 2)
VALIDATION_FRACTION = value_range(0.01, 0.5, 0.01, 3)
N_ITER_NO_CHANGE = value_range(5, 50, 5)
LEARNING_RATE = value_range(0.01, 1.0, 0.01, 3)
ALPHA = value_range(0.000001, 0.001, 0.00001, 8)
NN_WIDTH = value_range(10, 1000, 10)
NN_LEARNING_RATE = value_range(0.0001, 0.01, 0.001, 5)
NN_ARCHITECTURES = [(w,) for w in NN_WIDTH] + [(w, w) for w in NN_WIDTH]
KNN_NEIGHBORS = value_range(3, 101, 2)
LOGIT_C = [10 ** e for e in value_range(-4.0, 2.0, 0.25, 2)]
L1_RATIO = value_range(0.0, 1.0, 0.05, 2)

# Each algorithm keeps its own n_jobs at 1. The parallelism is spent on the
# search, where there are far more independent fits to spread across cores.
ALGORITHMS = {
    'Penalised logistic regression': {
        'slug': 'logisticregression',
        'estimator': LogisticRegression(penalty='elasticnet', solver='saga',
                                        max_iter=5000, random_state=RANDOM_STATE, n_jobs=1),
        'params': {
            'clf__C': LOGIT_C,
            'clf__l1_ratio': L1_RATIO,
        },
    },
    'Random forest': {
        'slug': 'randomforest',
        'estimator': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        'params': {
            'clf__n_estimators': N_ESTIMATORS,
            'clf__max_depth': FOREST_DEPTH,
        },
    },
    'Neural network': {
        'slug': 'neuralnetwork',
        'estimator': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
        'params': {
            'clf__hidden_layer_sizes': NN_ARCHITECTURES,
            'clf__learning_rate_init': NN_LEARNING_RATE,
            'clf__alpha': ALPHA,
            # Stopping on a held out slice or on the training loss are different
            # models, so the choice is tuned rather than fixed.
            'clf__early_stopping': [True, False],
        },
    },
    'XGBoost': {
        'slug': 'xgboost',
        'estimator': XGBClassifier(random_state=RANDOM_STATE, n_jobs=1,
                                   tree_method='hist', eval_metric='logloss'),
        'params': {
            'clf__n_estimators': N_ESTIMATORS,
            'clf__max_depth': BOOSTED_DEPTH,
            'clf__subsample': SUBSAMPLE,
        },
    },
    'Gradient boosting': {
        'slug': 'gradientboosting',
        'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'clf__n_estimators': N_ESTIMATORS,
            'clf__max_features': MAX_FEATURES,
            'clf__subsample': SUBSAMPLE,
            'clf__validation_fraction': VALIDATION_FRACTION,
            'clf__n_iter_no_change': N_ITER_NO_CHANGE,
        },
    },
    'Histogram gradient boosting': {
        'slug': 'histgradientboosting',
        'estimator': HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                                    early_stopping=True),
        'params': {
            'clf__max_depth': BOOSTED_DEPTH,
            'clf__learning_rate': LEARNING_RATE,
            'clf__validation_fraction': VALIDATION_FRACTION,
            'clf__n_iter_no_change': N_ITER_NO_CHANGE,
        },
    },
    'Bagging': {
        'slug': 'bagging',
        'estimator': BaggingClassifier(random_state=RANDOM_STATE, n_jobs=1),
        'params': {
            'clf__n_estimators': N_ESTIMATORS,
            'clf__max_features': MAX_FEATURES,
            'clf__max_samples': SUBSAMPLE,
        },
    },
    'SGD classifier': {
        'slug': 'sgdclassifier',
        'estimator': SGDClassifier(random_state=RANDOM_STATE, max_iter=1000),
        'params': {
            # Only these two losses give a probability estimate. The others fit a
            # decision boundary with no probability attached, and neither AUC nor
            # calibration could be computed from them.
            'clf__loss': ['log_loss', 'modified_huber'],
            'clf__alpha': ALPHA,
            'clf__early_stopping': [True, False],
            'clf__validation_fraction': VALIDATION_FRACTION,
            'clf__n_iter_no_change': N_ITER_NO_CHANGE,
        },
    },
    'K-nearest neighbors': {
        'slug': 'kneighbors',
        'estimator': KNeighborsClassifier(n_jobs=1),
        'params': {
            'clf__n_neighbors': KNN_NEIGHBORS,
            'clf__weights': ['uniform', 'distance'],
            'clf__p': [1, 2],
        },
    },
}
