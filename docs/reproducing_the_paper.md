# Reproducing the paper

Which command produces which table and figure, and what to expect from each.

## Without data access

Everything reported in the paper is regenerated from `results/all_results.csv`,
which is committed. No data use agreement is needed for any of this.

| Item | Command | Output |
| --- | --- | --- |
| Table 1, the eight tasks | `python scripts/04_make_tables.py` | `tables/table1_tasks.csv` |
| Table 2, search spaces | same | `tables/table2_search_spaces.csv` |
| Table 3, test AUC | same | `tables/table3_auc.csv` |
| Table 4, calibration slope | same | `tables/table4_calibration.csv` |
| Table 5, algorithm summary | same | `tables/table5_summary.csv` |
| Table 6, selection rules | same | `tables/table6_selection_rule.csv` |
| Figure 1, design | `python scripts/05_make_figures.py` | `figures/figure1.png`, `.pdf` |
| Figure 2, AUC against calibration | same | `figures/figure2.png`, `.pdf` |

Table 2 is built from the algorithm definitions in `src/benchmark/algorithms.py`
rather than from results, so the reported search space and the searched space are
the same object.

`scripts/03_compare_selection_rules.py` regenerates
`results/selection_rule_comparison.csv` and `results/cv_selected_models.csv`, and
prints the comparison against penalised logistic regression.

## Numbers to check

If a change has broken something, these are the values that will move first.

- 72 models, 8 tasks, 9 algorithms
- Test AUC from 0.527 to 0.825
- Calibration slope from 0.02 to 1.43
- 28 of 72 models have a calibration slope between 0.80 and 1.20
- Penalised logistic regression has the highest test AUC in 4 of 8 tasks
- The two selection rules disagree in 4 of 8 tasks
- Mean AUC gain from selecting on the test set is 0.003

`pytest -q` asserts all of these.

## With data access

```bash
python scripts/01_build_matrices.py --women <women.csv> --men <men.csv>
python scripts/02_run_benchmark.py
```

Step 1 writes the matrices and `results/split_summary.csv`, then runs four checks
and stops if any fails rather than letting a bad matrix reach a model.

Step 2 is 36,000 fits. Expect hours rather than minutes. It checkpoints after
every algorithm. Six checks run at the end.

Both scripts take `--outdir` so a trial run can be pointed away from `results/`.

## Figure sizes

Both figures are generated at exactly 6.5 inches wide, which is the text column
of a letter page with one inch margins. Do not rescale them when placing them,
or the font sizes will no longer match the specification.
