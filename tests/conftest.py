import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def results():
    return pd.read_csv(ROOT / 'results' / 'all_results.csv')


@pytest.fixture(scope='session')
def split_summary():
    return pd.read_csv(ROOT / 'results' / 'split_summary.csv')


@pytest.fixture(scope='session')
def domain_map():
    with open(ROOT / 'data' / 'schema' / 'domain_map.json') as f:
        return json.load(f)
