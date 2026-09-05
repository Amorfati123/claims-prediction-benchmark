"""The two figures in the paper.

Figure 2 is built entirely from results/all_results.csv, so the plotted values
are the same values reported in the tables. The checks below run before anything
is written; if one of them fails the figure is wrong and nothing is saved.
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config as cfg

RESULTS = cfg.RESULTS_DIR / 'all_results.csv'
OUTDIR = cfg.FIGURES_DIR

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
})

DISPLAY = {m: cfg.display(m) for m in cfg.ALGORITHM_ORDER}

# Okabe-Ito derived palette, chosen to separate in greyscale as well as colour.
# Marker shape carries the same information as colour so the figure survives a
# black and white printout.
STYLE = {
    'Penalised logistic regression': ('#0072B2', 'o'),
    'Gradient boosting': ('#009E73', 's'),
    'Histogram gradient boosting': ('#56B4E9', '^'),
    'Bagging': ('#CC79A7', 'D'),
    'XGBoost': ('#D55E00', 'v'),
    'Random forest': ('#E69F00', 'P'),
    'SGD classifier': ('#7A5195', 'X'),
    'Neural network': ('#666666', '<'),
    'K-nearest neighbors': ('#000000', '*'),
}

# Models selected on cross-validated AUC in each task, from stepBC_best_models.csv.
CV_SELECTED = {
    'T1': 'Gradient boosting', 'T2': 'Gradient boosting',
    'T3': 'Gradient boosting', 'T4': 'Gradient boosting',
    'T5': 'Gradient boosting', 'T6': 'Penalised logistic regression',
    'T7': 'Penalised logistic regression', 'T8': 'Gradient boosting',
}

# The four models discussed by name in the Results.
CALLOUTS = [
    (1, 'Penalised logistic regression', 'T1'),
    (2, 'XGBoost', 'T2'),
    (3, 'Gradient boosting', 'T8'),
    (4, 'Random forest', 'T8'),
]


def load():
    df = cfg.add_task_column(pd.read_csv(RESULTS))
    df = df.rename(columns={'Calibration slope': 'slope'})
    return df[['Task', 'Model', 'AUC', 'slope']]


def checks(df):
    """Print the correctness checks. Every expected value here was verified
    independently against the results file before the figure was drawn."""
    band = df[(df.slope >= 0.80) & (df.slope <= 1.20)]
    results = [
        ('points plotted is 72', len(df) == 72, len(df)),
        ('AUC spans 0.527 to 0.825',
         round(df.AUC.min(), 3) == 0.527 and round(df.AUC.max(), 3) == 0.825,
         (round(df.AUC.min(), 3), round(df.AUC.max(), 3))),
        ('slope spans 0.02 to 1.43',
         round(df.slope.min(), 2) == 0.02 and round(df.slope.max(), 2) == 1.43,
         (round(df.slope.min(), 2), round(df.slope.max(), 2))),
        ('28 points inside the 0.80 to 1.20 band', len(band) == 28, len(band)),
        # Bounds are compared at the two decimal places the manuscript tables
        # report. The smallest random forest slope is 0.9954, which is 1.00 as
        # displayed, so comparing raw values against the printed bound fails.
        ('XGBoost slopes within 0.17 to 0.44 as displayed',
         df[df.Model == 'XGBoost'].slope.round(2).between(0.17, 0.44).all(), None),
        ('Random forest slopes within 1.00 to 1.43 as displayed',
         df[df.Model == 'Random forest'].slope.round(2).between(1.00, 1.43).all(), None),
        ('all k-nearest neighbours AUC below 0.60',
         (df[df.Model == 'K-nearest neighbors'].AUC < 0.60).all(), None),
    ]
    ok = True
    for label, passed, observed in results:
        mark = 'pass' if passed else 'FAIL'
        extra = '' if observed is None else f'  (observed {observed})'
        print(f'  [{mark}] {label}{extra}')
        ok = ok and passed
    return ok


def figure2(df):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    ax.axhspan(0.80, 1.20, color='0.90', zorder=0)
    ax.axhline(1.00, color='0.35', lw=0.9, zorder=1)
    ax.text(0.505, 1.012, 'ideal calibration', ha='left', va='bottom',
            fontsize=6.2, color='0.35', zorder=1)

    for model, (colour, marker) in STYLE.items():
        sub = df[df.Model == model]
        ax.scatter(sub.AUC, sub.slope, s=24, c=colour, marker=marker,
                   linewidths=0.4, edgecolors='white', zorder=3)

    # Ring the model that the prespecified rule would report in each task.
    sel = df[[m == CV_SELECTED[t] for t, m in zip(df.Task, df.Model)]]
    ax.scatter(sel.AUC, sel.slope, s=88, facecolors='none', edgecolors='black',
               linewidths=0.8, zorder=4)

    # The upper right of the cloud is dense, so each callout gets a short
    # leader line rather than a bare offset label.
    label_at = {1: (0.8265, 1.345), 2: (0.8215, 0.225),
                3: (0.6205, 0.700), 4: (0.7060, 1.470)}
    for number, model, task in CALLOUTS:
        row = df[(df.Model == model) & (df.Task == task)].iloc[0]
        colour, marker = STYLE[model]
        ax.scatter([row.AUC], [row.slope], s=34, c=colour, marker=marker,
                   linewidths=0.6, edgecolors='black', zorder=5)
        ax.annotate(str(number), xy=(row.AUC, row.slope),
                    xytext=label_at[number], textcoords='data',
                    fontsize=7.5, fontweight='bold', ha='center', va='center',
                    arrowprops=dict(arrowstyle='-|>', lw=0.6, color='0.25',
                                    shrinkA=4, shrinkB=3, mutation_scale=6),
                    zorder=6)

    # The upper left of the plotting area contains no models, so the key sits
    # inside the axes rather than stealing height from the figure.
    key_rows = []
    for n, m, t in CALLOUTS:
        r = df[(df.Model == m) & (df.Task == t)].iloc[0]
        key_rows.append(f'{n}  {DISPLAY[m]}, {t}  ({r.AUC:.3f}, {r.slope:.2f})')
    key = '\n'.join(key_rows)
    ax.text(0.504, 1.475, key, fontsize=5.9, va='top', ha='left',
            linespacing=1.5, zorder=5)

    ax.set_xlim(0.50, 0.85)
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel('Test AUC', fontsize=9)
    ax.set_ylabel('Calibration slope', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, color='0.88', zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], linestyle='none',
                      markersize=4.5, markeredgecolor='white', markeredgewidth=0.4,
                      label=DISPLAY[m]) for m in STYLE]
    handles.append(Line2D([], [], color='black', marker='o', linestyle='none',
                          markersize=7, markerfacecolor='none', markeredgewidth=0.8,
                          label='selected on cross-validated AUC'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.03),
              fontsize=6.3, frameon=False, handletextpad=0.5, labelspacing=0.52,
              borderaxespad=0)

    # Reserve the right hand strip for the legend inside the 6.5 inch canvas.
    fig.subplots_adjust(left=0.083, right=0.655, bottom=0.165, top=0.975)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / 'figure2.png', dpi=300)
    fig.savefig(OUTDIR / 'figure2.pdf')
    plt.close(fig)


def figure1():
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    # The axes fill the canvas so the saved file is exactly 6.5 x 1.5 inches.
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.13, 1.03)
    ax.axis('off')

    boxes = [
        ('Cohort',
         'Medicare 5% LDS, 2013 to 2020\n'
         '11,049 adults hospitalized with\nischemic stroke\n'
         '576 candidate predictors\nacross 5 domains'),
        ('8 prediction tasks',
         '3 outcomes x 2 strata\n(full predictor set, 574)\n'
         '+ combined outcome x 2 strata\n(discharge-only set, 570)'),
        ('9 algorithms per task',
         'penalised logistic regression,\ngradient boosting, histogram\n'
         'gradient boosting, XGBoost,\nbagging, random forest, SGD,\n'
         'neural network, k-NN'),
        ('Evaluation of 72 models',
         'discrimination (AUC)\ncalibration slope and intercept\n'
         'Brier score\n1,000-sample bootstrap intervals'),
    ]

    left, width, gap = 0.006, 0.2245, 0.0273
    top, bottom = 1.00, 0.30
    for i, (header, body) in enumerate(boxes):
        x0 = left + i * (width + gap)
        ax.add_patch(FancyBboxPatch(
            (x0, bottom), width, top - bottom,
            boxstyle='round,pad=0.004,rounding_size=0.012',
            linewidth=0.7, edgecolor='#333333', facecolor='#F5F5F5', zorder=2))
        ax.text(x0 + width / 2, top - 0.07, header, ha='center', va='top',
                fontsize=7.4, fontweight='bold', zorder=3)
        ax.text(x0 + width / 2, top - 0.20, body, ha='center', va='top',
                fontsize=5.9, linespacing=1.32, zorder=3)
        if i < 3:
            ax.add_patch(FancyArrowPatch(
                (x0 + width + 0.0015, (top + bottom) / 2),
                (x0 + width + gap - 0.0015, (top + bottom) / 2),
                arrowstyle='-|>', mutation_scale=9, linewidth=1.0,
                color='#333333', zorder=3))

    # Bracket spanning the two stages that are held constant across algorithms.
    b0 = left + 1 * (width + gap)
    b1 = left + 2 * (width + gap) + width
    ax.plot([b0, b0, b1, b1], [0.22, 0.16, 0.16, 0.22],
            color='#333333', lw=0.7, zorder=3)
    ax.text((b0 + b1) / 2, 0.125,
            'identical partition, identical preprocessing, identical tuning budget',
            ha='center', va='top', fontsize=6.2, style='italic', zorder=3)

    ax.add_patch(FancyBboxPatch(
        (left, -0.10), 1 - 2 * left, 0.115,
        boxstyle='round,pad=0.002,rounding_size=0.010',
        linewidth=0.7, edgecolor='#333333', facecolor='#E8E8E8', zorder=2))
    ax.text(0.5, -0.043,
            '8 tasks          9 algorithms          72 models          36,000 fits',
            ha='center', va='center', fontsize=6.8, fontweight='bold', zorder=3)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / 'figure1.png', dpi=300)
    fig.savefig(OUTDIR / 'figure1.pdf')
    plt.close(fig)


def write_all():
    data = load()
    print('Figure 2 correctness checks:')
    if not checks(data):
        raise SystemExit('a correctness check failed, nothing written')
    figure2(data)
    figure1()
    print('\nwrote figure1.png, figure1.pdf, figure2.png, figure2.pdf')
