# Data

No patient-level data is in this repository and none will be added.

## The cohort

The study uses a 5% random sample of Medicare fee-for-service beneficiaries with
Part D coverage, covering 2013 to 2020. It is obtained from the Centers for
Medicare and Medicaid Services under a data use agreement and cannot be
redistributed. We cannot supply it on request. Cohort construction is described
in the Methods section of the paper.

## What the pipeline expects

Two prepared files, one per stratum, each containing:

- every column listed in `schema/predictor_manifest.csv`
- an `ID` column, kept alongside the matrices for traceability and never used as
  a predictor
- the three outcome columns named in `configs/benchmark.yaml`, which are the
  binary non-adherence indicators at the 0.80 threshold

Missing values should already be resolved before this point. The three
structurally missing adherence measures are filled with zero and paired with an
indicator recording that the measure did not apply, which is how the published
run handled them.

## What is in schema/

`predictor_manifest.csv` lists all 590 columns and the domain each belongs to.
Names and domains only, no values. Of these, 576 are candidate predictors across
the five predictor domains, 12 are adherence-derived columns excluded from every
predictor set, and 2 are identifiers.

`domain_map.json` is the same mapping in the form the code reads.

`domain_counts.csv` is the count of candidate predictors per domain.

## Running without the data

`synthetic/make_synthetic_cohort.py` writes stand-in files with the same column
names, the same types and roughly the same marginal distributions. It builds in
enough signal that the algorithms produce something other than an AUC of 0.5, so
a run tells you whether the pipeline works.

It will not reproduce any published number and is not meant to.
