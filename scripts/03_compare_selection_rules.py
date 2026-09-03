"""Compare the two model selection rules.

The rule used in the paper picks the model with the highest cross-validated AUC.
The alternative picks the one with the highest test AUC. The point of this script
is the second column of the output rather than the first: the gain in reported
AUC is negligible, but which model gets reported changes in half the tasks.
"""

import pandas as pd

from benchmark import config as cfg
from benchmark.selection import compare_rules, compare_with_logistic, select_on_cv


def main():
    results = pd.read_csv(cfg.RESULTS_DIR / 'all_results.csv')

    comparison = compare_rules(results)
    comparison.to_csv(cfg.RESULTS_DIR / 'selection_rule_comparison.csv', index=False)

    print(comparison[['Task', 'Model selected on CV', 'CV rule test AUC',
                      'CV rule calibration slope', 'Model selected on test',
                      'Test rule test AUC', 'Test rule calibration slope',
                      'AUC gain from test selection']].to_string(index=False))
    print()
    print(f'Mean AUC gain from selecting on the test set: '
          f'{comparison["AUC gain from test selection"].mean():.4f}')
    print(f'Tasks where the two rules report a different model: '
          f'{int(comparison["Rules disagree"].sum())} of {len(comparison)}')

    selected = select_on_cv(results)
    selected.to_csv(cfg.RESULTS_DIR / 'cv_selected_models.csv', index=False)

    print('\nSelected model against penalised logistic regression\n')
    versus = compare_with_logistic(results, selected)
    print(versus.round(3).to_string(index=False))
    print(f'\nMean test AUC gain over penalised logistic regression: '
          f'{versus["AUC gain"].mean():.4f}')


if __name__ == '__main__':
    main()
