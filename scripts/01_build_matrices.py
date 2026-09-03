"""Build the modelling matrices and splits.

Needs the two prepared cohort files, which are not in this repository. Point the
arguments at your own copies, or run synthetic/make_synthetic_cohort.py first to
generate stand-ins that let the pipeline run end to end.
"""

import argparse
import json
from pathlib import Path

from benchmark import config as cfg
from benchmark import data as bd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--women', required=True, help='prepared cohort file for women')
    parser.add_argument('--men', required=True, help='prepared cohort file for men')
    parser.add_argument('--outdir', default=None,
                        help='where to write split_summary.csv, defaults to results/. '
                             'Point it elsewhere for a synthetic run so the published '
                             'results are not overwritten.')
    args = parser.parse_args()

    cfg.ensure_directories()
    outdir = Path(args.outdir) if args.outdir else cfg.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    frames = bd.load_cohort(args.women, args.men)
    domain_map = bd.load_domain_map()

    for sex, d in frames.items():
        print(f'{sex}: {d.shape[0]:,} patients, {d.shape[1]} columns, '
              f'{int(d.isna().sum().sum())} missing values')

    configurations, split_summary, constants = bd.build_matrices(frames, domain_map)
    excluded = bd.adherence_derived(domain_map)

    for sex, cols in constants.items():
        print(f'{sex}: constant columns removed {cols}')
    print(f'\nAdherence derived columns excluded from every predictor set: {len(excluded)}')

    split_summary.to_csv(outdir / 'split_summary.csv', index=False)
    with open(cfg.MATRIX_DIR / 'manifest.json', 'w') as f:
        json.dump(configurations, f, indent=2)

    print()
    print(split_summary.to_string(index=False))

    print('\nChecks')
    report = bd.check_matrices(configurations, split_summary, excluded)
    for label, passed in report.items():
        print(f'  [{"pass" if passed else "FAIL"}] {label}')
    if not all(report.values()):
        raise SystemExit('a check failed, stopping before any model is fitted')


if __name__ == '__main__':
    main()
