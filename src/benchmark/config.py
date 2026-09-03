"""Settings for the benchmark.

Everything that changes a result lives here or in configs/benchmark.yaml, so a
reader can see the whole set of choices in one place rather than hunting through
the code for a hard coded number.
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / 'configs'
RESULTS_DIR = ROOT / 'results'
TABLES_DIR = ROOT / 'tables'
FIGURES_DIR = ROOT / 'figures'
SCHEMA_DIR = ROOT / 'data' / 'schema'


def load(name='benchmark.yaml'):
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


CONFIG = load()

TEST_SIZE = CONFIG['split']['test_size']
RANDOM_STATE = CONFIG['split']['random_state']
CV_FOLDS = CONFIG['tuning']['cv_folds']
N_SEARCH_ITER = CONFIG['tuning']['n_search_iter']
TUNING_METRIC = CONFIG['tuning']['metric']
N_BOOTSTRAP = CONFIG['evaluation']['n_bootstrap']
CALIBRATION_BINS = CONFIG['evaluation']['calibration_bins']

OUTCOMES = CONFIG['outcomes']
VARIANTS = CONFIG['variants']
BASELINE_OUTCOMES = CONFIG['baseline_outcomes']
POST_INDEX_PREDICTORS = CONFIG['variants']['Baseline']
PREDICTOR_DOMAINS = set(CONFIG['predictor_domains'])

MATRIX_DIR = ROOT / CONFIG['paths']['matrix_dir']
MODEL_DIR = ROOT / CONFIG['paths']['model_dir']
CHECKPOINT_DIR = ROOT / CONFIG['paths']['checkpoint_dir']
DOMAIN_MAP_PATH = ROOT / CONFIG['paths']['domain_map']

# On a cluster the scheduler tells us how many cores we actually have. Locally
# joblib works it out itself.
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))

# Short task identifiers used in the paper's tables and figures.
TASK_IDS = {
    ('Combined', 'PostIndex', 'Women'): 'T1',
    ('Combined', 'PostIndex', 'Men'): 'T2',
    ('Statins', 'PostIndex', 'Women'): 'T3',
    ('Statins', 'PostIndex', 'Men'): 'T4',
    ('Antihypertensives', 'PostIndex', 'Women'): 'T5',
    ('Antihypertensives', 'PostIndex', 'Men'): 'T6',
    ('Combined', 'Baseline', 'Women'): 'T7',
    ('Combined', 'Baseline', 'Men'): 'T8',
}

# The source data spells this one the American way. The paper uses the British
# spelling, so the mapping lives here rather than being applied ad hoc.
DISPLAY_NAMES = {'K-nearest neighbors': 'K-nearest neighbours'}

ALGORITHM_ORDER = [
    'Penalised logistic regression',
    'Gradient boosting',
    'Histogram gradient boosting',
    'Bagging',
    'XGBoost',
    'Random forest',
    'SGD classifier',
    'Neural network',
    'K-nearest neighbors',
]


def task_id(outcome, variant, sex):
    return TASK_IDS[(outcome, variant, sex)]


def add_task_column(frame):
    """Attach the short task identifier to any results frame."""
    frame = frame.copy()
    frame['Task'] = [task_id(o, v, s)
                     for o, v, s in zip(frame.Outcome, frame.Variant, frame.Sex)]
    return frame


def display(model):
    return DISPLAY_NAMES.get(model, model)


def ensure_directories():
    for directory in (MATRIX_DIR, MODEL_DIR, CHECKPOINT_DIR,
                      RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
