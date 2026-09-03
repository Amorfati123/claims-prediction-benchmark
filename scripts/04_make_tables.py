"""Regenerate every table in the paper."""

import pandas as pd

from benchmark import config as cfg
from benchmark.tables import write_all


def main():
    results = pd.read_csv(cfg.RESULTS_DIR / 'all_results.csv')
    split_summary = pd.read_csv(cfg.RESULTS_DIR / 'split_summary.csv')

    tables = write_all(results, split_summary)
    for name, frame in tables.items():
        print(f'\n{name}  ({len(frame)} rows)')
        print(frame.to_string(index=False))
    print(f'\nWritten to {cfg.TABLES_DIR}')


if __name__ == '__main__':
    main()
