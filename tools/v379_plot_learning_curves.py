"""Plot completed v379 development curves only; never fits or reselects."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TITLES = ('Caco2', 'LD50', 'Lipophilicity', 'PPBR', 'Solubility',
          'Hepatocyte clearance*', 'Microsome clearance*', 'Half-life*', 'VDss*')


def main():
    evidence = json.loads((ROOT / 'docs/v379_learning_curves.json').read_text())
    tasks = list(dict.fromkeys(r['task'] for r in evidence['records']))
    assert len(tasks) == 9
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.subplots_adjust(left=.08, right=.985, top=.925, bottom=.18, hspace=.4, wspace=.28)
    labels = dict(small_ema_l1='Small Transformer + EMA + L1',
                  small_ema_quantiles='Small Transformer + EMA + quantiles')
    for ax, task, title in zip(axes.flat, tasks, TITLES):
        for record in evidence['records']:
            if record['task'] != task:
                continue
            curve = record['curve']
            ax.plot([r['rounds'] for r in curve], [100*r['mean_relative_MAE_gain'] for r in curve],
                    marker='o', label=labels[record['base']])
        ax.axhline(0, linewidth=.7, color='#777777')
        ax.axhline(5, linewidth=.7, color='#aaaaaa', linestyle='--')
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_xticks((1000, 3000, 6000), ('1000', '3000', '6000'))
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('v379 complete learning curves: fixed blend weight per method', fontsize=14)
    fig.supylabel('Relative MAE reduction versus reference (%)', fontsize=11)
    fig.supxlabel('Training steps per seed (both seeds completed 6000 steps)', y=.13, fontsize=10)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .07), ncol=2, frameon=False)
    fig.text(.5, .015, 'Mean of fold0/1. Development-selected weights held fixed along each curve.\n'
             '* Calibrated historical-score MAE reference; official Spearman checked separately.\n'
             'Descriptive analysis only; no selection using confirmation results.', ha='center', fontsize=9)
    path = ROOT / 'docs/v379_learning_curves.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
