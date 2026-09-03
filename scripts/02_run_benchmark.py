"""Train and evaluate every algorithm on every task.

This is the long one. Nine algorithms across eight tasks, each tuned by a
hundred sampled parameter combinations under five-fold cross validation, is
36,000 fits. Results are checkpointed after each algorithm, so an interrupted
run picks up where it stopped.
"""

import argparse
import json
from pathlib import Path

from benchmark import config as cfg
from benchmark import data as bd
from benchmark import train as bt
from benchmark.algorithms import ALGORITHMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', action='store_true',
                        help='ignore existing checkpoints and refit everything')
    parser.add_argument('--no-save-models', action='store_true',
                        help='skip writing fitted models to disk')
    parser.add_argument('--n-iter', type=int, default=None,
                        help='override the number of sampled parameter combinations, '
                             'for a quick check that the pipeline runs')
    parser.add_argument('--bootstrap', type=int, default=None,
                        help='override the number of bootstrap resamples')
    parser.add_argument('--outdir', default=None,
                        help='where to write the result files, defaults to results/. '
                             'Point it elsewhere for a synthetic run so the published '
                             'results are not overwritten.')
    args = parser.parse_args()

    # Overrides exist so the pipeline can be exercised quickly. Anything other
    # than the configured values produces results that are not comparable with
    # the published ones, so it says so.
    if args.n_iter:
        cfg.N_SEARCH_ITER = args.n_iter
        print(f'search reduced to {args.n_iter} combinations, results are not publication runs')
    if args.bootstrap:
        cfg.N_BOOTSTRAP = args.bootstrap
        import benchmark.metrics as bm
        bm.N_BOOTSTRAP = args.bootstrap

    cfg.ensure_directories()
    outdir = Path(args.outdir) if args.outdir else cfg.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    with open(cfg.MATRIX_DIR / 'manifest.json') as f:
        configurations = json.load(f)

    data = bd.load_matrices(configurations)
    print(f'{len(data)} configurations loaded and verified against the manifest')
    print(f'{len(ALGORITHMS)} algorithms per configuration, '
          f'{len(ALGORITHMS) * len(data)} models in total\n')

    bt.run(configurations, data, resume=not args.fresh,
           save_model=not args.no_save_models)

    results, hyperparameters, curves = bt.collect(configurations)
    results.to_csv(outdir / 'all_results.csv', index=False)
    hyperparameters.to_csv(outdir / 'best_hyperparameters.csv', index=False)
    curves.to_csv(outdir / 'calibration_curves.csv', index=False)

    print(f'Saved all_results.csv ({len(results)} rows)')
    print(f'Saved best_hyperparameters.csv ({len(hyperparameters)} rows)')
    print(f'Saved calibration_curves.csv ({len(curves)} rows)')

    print('\nChecks')
    report = bt.check_results(results, configurations, len(ALGORITHMS))
    for label, passed in report.items():
        print(f'  [{"pass" if passed else "FAIL"}] {label}')
    if not all(report.values()):
        raise SystemExit('a check failed, results should not be reported as they stand')


if __name__ == '__main__':
    main()
