# Algorithm choice and model selection in claims-based clinical prediction

Code and aggregate results for a controlled benchmark of nine algorithms across eight prediction tasks built on one Medicare claims cohort. Everything except the algorithm and the selection rule is held fixed, so the differences that show up are attributable to those two choices and not to the data each model happened to see.

The short version of what we found. Discrimination is nearly flat across algorithms: on average seven of nine have confidence intervals overlapping the best, and penalised logistic regression has the highest test AUC in four of the eight tasks. Calibration is not flat at all. Among the six best-discriminating algorithms in a task, test AUC spans 0.03 to 0.06 while the calibration slope spans 0.80 to 1.22. Ranking models on discrimination alone will sometimes hand you a model whose predicted probabilities cannot be read as risks.

## Paper

**[CITATION AND DOI ONCE ACCEPTED]**

## What is and is not in here

The cohort comes from the Medicare 5% Limited Data Set, which cannot be redistributed. No patient-level data is in this repository and none will be added.

What is here: all of the code, the configuration that produced the published run, and the aggregate result files. `results/all_results.csv` is the benchmark, 72 rows of metrics with no patient information in it. Every table and both figures in the paper are regenerated from that file by the scripts below.

What is not here: the cohort, the prepared modelling matrices, and the fitted models. Fitted models are excluded deliberately as well as practically, since a boosted ensemble trained on 4,632 patients across 574 predictors can leak information about the rows it was trained on.

To run the pipeline without the real data, `synthetic/make_synthetic_cohort.py` writes stand-in files with the same shape and column names. It exercises every step and reproduces none of the published numbers, which is the point of it.

## Repository layout

```
configs/          settings that change a result, including the seed
data/schema/      predictor manifest and domain map, names only, no values
src/benchmark/    the package
scripts/          numbered drivers, run in order
synthetic/        stand-in cohort generator
results/          aggregate outputs from the published run
tables/           regenerated, not committed
figures/          regenerated, not committed
tests/            leakage, metric and selection rule checks
docs/             reproduction notes and the TRIPOD+AI checklist
```

## Installation

```bash
git clone https://github.com/Amorfati123/claims-prediction-benchmark.git
cd claims-prediction-benchmark
pip install -e ".[dev]"
```

or with conda:

```bash
conda env create -f environment.yml
conda activate claims-benchmark
```

Python 3.10 or newer.

## Data access

The Medicare 5% Limited Data Set is obtained from the Centers for Medicare and Medicaid Services under a data use agreement. It is not ours to share and it is not available on request from us. Cohort construction is described in the paper's Methods.

The pipeline expects two prepared files, one per stratum, each carrying the columns listed in `data/schema/predictor_manifest.csv` plus an `ID` column and the three outcome columns. `data/README.md` has the details.

## Reproducing the published results

The tables and figures come straight from the committed results, so this needs no data access and takes seconds:

```bash
python scripts/03_compare_selection_rules.py
python scripts/04_make_tables.py
python scripts/05_make_figures.py
```

`scripts/04` writes Tables 1 and 3 through 6 to `tables/`. Table 2 is the search space and is written from the algorithm definitions rather than from results, so the table cannot drift away from the code. `scripts/05` writes both figures to `figures/` as 300 dpi PNG and vector PDF, and refuses to write anything if its correctness checks fail.

## Running the benchmark from scratch

With the real data:

```bash
python scripts/01_build_matrices.py --women path/to/women.csv --men path/to/men.csv
python scripts/02_run_benchmark.py
python scripts/03_compare_selection_rules.py
python scripts/04_make_tables.py
python scripts/05_make_figures.py
```

Step 2 is the long one. Nine algorithms across eight tasks, each tuned over a hundred sampled parameter combinations under five-fold cross validation, is 36,000 fits. It checkpoints after every algorithm, so an interrupted run picks up where it stopped. Pass `--fresh` to ignore the checkpoints and start over.

Without the real data:

```bash
make smoke
```

That generates a small synthetic cohort, runs every algorithm on every task with a cut-down search, and writes to `work/results/` so it cannot overwrite anything published.

## Expected outputs

`results/all_results.csv` has 72 rows and 50 columns. Test AUC runs from 0.527 to 0.825 and the calibration slope from 0.02 to 1.43. Twenty-eight of the 72 models have a calibration slope within 0.2 of ideal. The two selection rules disagree in four of the eight tasks while differing by a mean of 0.003 AUC. `tests/test_selection.py` asserts each of these, so a change that breaks one will fail CI rather than quietly propagate.

## Reproducibility

One seed, set in `configs/benchmark.yaml`, governs the train and test split, the cross validation folds, the randomised search, the stochastic components of every algorithm, and the bootstrap resampling.

Two things matter more than the seed. Scaling happens inside the model pipeline, so the scaler is refitted within each cross validation fold and no fold contributes to its own standardisation. And the model reported for each task is chosen on cross-validated AUC, fixed before the test set is examined, so the test AUC we report is not the quantity that picked the model. `select_on_test` exists in `src/benchmark/selection.py` only so the two rules can be compared, and is never used to report performance.

Results were produced with the versions pinned in `requirements.txt`. Different versions of scikit-learn or XGBoost may shift results slightly.

## Tests

```bash
pytest -q
```

`tests/test_no_leakage.py` is the one worth reading first. It asserts that the scaler sits inside the pipeline, that every search space targets the classifier step, that all twelve adherence-derived columns are absent from every predictor set, that the discharge-only variant drops exactly the four post-discharge measures, and that the predictor counts in the results match the manifest. For a paper arguing that methodological discipline matters, a passing test proving our own pipeline is clean seemed like the least we could do.

## Limitations

One cohort, one payer, one country. The eight tasks share patients and most predictors, so they are eight views of one setting rather than eight replications. We did not apply post hoc recalibration such as Platt scaling or isotonic regression, which can substantially repair the calibration of algorithms like XGBoost; our results describe these algorithms as they are commonly used and reported, which is usually without it. The search budget is identical across algorithms by design, but algorithms differ in how efficiently they use one.

## License

MIT for the code, see `LICENSE`. The data is not ours to license.

## Citation

See `CITATION.cff`, or cite the paper directly once it appears.
