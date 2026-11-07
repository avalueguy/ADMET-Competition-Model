"""Post-selection descriptive analysis; never changes frozen model choices."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['377', '378', '379'], required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    exp = root / f'experiments/regression9_v{args.version}'
    development = json.loads((exp / 'outputs/development.json').read_text())
    records = []
    for record in development['records']:
        task = record['task']
        bases = sorted(set(r['config']['base'] for r in record['choices']))
        for name in bases:
            full = [r for r in record['choices'] if r['config']['base'] == name]
            horizons = sorted(set(r['config']['rounds'] for r in full))
            # Same weight across all horizons, chosen only using development.
            weight = next(r['weight'] for r in full if r['config']['rounds'] == max(horizons)
                          and r['objective'] == max(x['objective'] for x in full
                                                  if x['config']['rounds'] == max(horizons)))
            curve = []
            for length in horizons:
                row = next(r for r in full if r['weight'] == weight and r['config']['rounds'] == length)
                curve.append(dict(rounds=length, weight=weight,
                                  mean_relative_MAE_gain=float(np.mean([p['gain'] for p in row['parts']])),
                                  min_relative_MAE_gain=min(p['gain'] for p in row['parts']),
                                  rank_gains=[p.get('rank_gain') for p in row['parts']]))
            records.append(dict(task=task, base=name, curve=curve,
                                late_vs_early_mean_relative_gain=curve[-1]['mean_relative_MAE_gain'] - curve[0]['mean_relative_MAE_gain'],
                                late_better_on_both_development_parts=all(
                                    b['gain'] > a['gain'] for a, b in zip(
                                        next(r['parts'] for r in full if r['weight'] == weight and r['config']['rounds'] == min(horizons)),
                                        next(r['parts'] for r in full if r['weight'] == weight and r['config']['rounds'] == max(horizons))))))
    value = dict(version=f'v{args.version}', signature=development['signature'],
                 purpose='Descriptive fixed-weight learning curves; no confirmation retuning, no fitting.',
                 late_better_on_both_parts=sum(r['late_better_on_both_development_parts'] for r in records),
                 total_task_base_pairs=len(records), records=records)
    (root / f'docs/v{args.version}_learning_curves.json').write_text(
        json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k != 'records'}))


if __name__ == '__main__':
    main()
