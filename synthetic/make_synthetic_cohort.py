"""Generates stand-in cohort files so the pipeline can be run without the real data.

The Medicare Limited Data Set cannot be redistributed, so this writes two files
with the same column names, the same types and roughly the same marginal
distributions as the real prepared files. It is enough to exercise every step of
the pipeline and to check that the code runs, and it is not enough to reproduce
any number in the paper. Results from synthetic data will not match the results
in results/, and they are not meant to.

The generator does build in a little real structure: the outcome depends on the
adherence measures, and the post-discharge measures carry more of that signal
than the pre-stroke ones. Without it every algorithm would score an AUC of 0.5
and a run would tell you nothing about whether the pipeline works.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_MAP = ROOT / 'data' / 'schema' / 'domain_map.json'

# Sizes of the real strata, so the synthetic files exercise the same shapes.
N_WOMEN = 5790
N_MEN = 5259

# Columns the pipeline needs by name rather than by domain.
ADHERENCE_CONTINUOUS = [
    'statin_PDC', 'ANTIHYPERTENSIVE_PDC', 'statin_antihyp_pdc',
    'statin_PDC_3mon', 'ANTIHYPERTENSIVE_PDC_3mon', 'statin_antihyp_pdc_3mon',
    'pdc_year_prior', 'pdc_6mon_prior',
    'statin_pdc_yr_prior', 'antihypertensive_pdc_yr_prior',
]
BINARY_SUFFIXES = ('_ind', '_binary', '_80', '_60')
COUNT_COLUMNS = ('num_', 'AHRF_TOT', 'POS_TOT', 'NOAAS_TOT')

# Sex specific conditions, held constant within a stratum so the constant column
# check in the pipeline has something real to find.
FEMALE_ONLY = ['endometrial_cancer_ind', 'breast_cancer_ind']
MALE_ONLY = ['prostate_cancer_ind', 'hyperplasia_ind']


def column_kind(name, domain):
    if name in ADHERENCE_CONTINUOUS:
        return 'proportion'
    if name.endswith(BINARY_SUFFIXES) or name.startswith('no_'):
        return 'binary'
    if name.startswith(COUNT_COLUMNS):
        return 'count'
    if name.startswith('ACS_PCT') or name.endswith('_RATE'):
        return 'percentage'
    if domain == 'community SDOH':
        return 'continuous'
    return 'continuous'


def draw(rng, kind, n):
    if kind == 'proportion':
        return np.clip(rng.beta(4, 1.6, n), 0, 1)
    if kind == 'binary':
        return rng.binomial(1, rng.uniform(0.03, 0.45), n)
    if kind == 'count':
        return rng.poisson(rng.uniform(0.5, 12), n)
    if kind == 'percentage':
        return np.clip(rng.normal(rng.uniform(5, 60), 12, n), 0, 100)
    return rng.normal(rng.uniform(-2, 60), rng.uniform(1, 20), n)


def build_stratum(rng, columns, domain_map, n, sex_code, constant_columns):
    drawn = {}
    for name in columns:
        if name in constant_columns:
            drawn[name] = np.zeros(n, dtype=int)
        else:
            drawn[name] = draw(rng, column_kind(name, domain_map.get(name)), n)

    drawn['SEX_IDENT_CD'] = np.full(n, sex_code)
    drawn['age_at_index'] = np.clip(rng.normal(78.6, 7.2, n), 65, 100)
    drawn['ID'] = np.arange(1, n + 1) + (0 if sex_code == 2 else 1_000_000)

    frame = pd.DataFrame(drawn)

    # A signal worth predicting. Early refill behaviour carries most of it, the
    # pre-stroke trajectory carries some, everything else is noise.
    linear = (-4.2
              + 4.6 * (1 - frame['statin_antihyp_pdc_3mon'])
              + 1.4 * (1 - frame['statin_pdc_yr_prior'])
              + 0.9 * (1 - frame['antihypertensive_pdc_yr_prior'])
              + 0.4 * rng.normal(0, 1, n))
    probability = 1 / (1 + np.exp(-linear))
    frame['class'] = rng.binomial(1, probability)

    # The two class specific outcomes track the combined one but are rarer, which
    # reproduces the prevalence ordering of the real tasks.
    frame['pdc_statin_80'] = rng.binomial(1, np.clip(probability * 0.68, 0, 1))
    frame['pdc_ANTIHYPERTENSIVE_80'] = rng.binomial(1, np.clip(probability * 0.55, 0, 1))

    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=str(ROOT / 'work' / 'synthetic'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=float, default=1.0,
                        help='shrink the cohort for a fast smoke test, for example 0.1')
    args = parser.parse_args()

    with open(DOMAIN_MAP) as f:
        domain_map = json.load(f)

    columns = list(domain_map)
    rng = np.random.default_rng(args.seed)

    women = build_stratum(rng, columns, domain_map,
                          int(N_WOMEN * args.scale), 2, MALE_ONLY)
    men = build_stratum(rng, columns, domain_map,
                        int(N_MEN * args.scale), 1, FEMALE_ONLY)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    women.to_csv(outdir / 'synthetic_women.csv', index=False)
    men.to_csv(outdir / 'synthetic_men.csv', index=False)

    print(f'women: {women.shape[0]:,} rows, {women.shape[1]} columns, '
          f'{100 * women["class"].mean():.1f}% non-adherent')
    print(f'men:   {men.shape[0]:,} rows, {men.shape[1]} columns, '
          f'{100 * men["class"].mean():.1f}% non-adherent')
    print(f'\nwritten to {outdir}')
    print('These are not real patients and will not reproduce the published results.')


if __name__ == '__main__':
    main()
