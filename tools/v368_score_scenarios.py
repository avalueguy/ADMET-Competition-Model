"""Conditional scoring arithmetic, never an official-score forecast."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def task_score(metric, value, reference):
    if metric == 'MAE':
        return 100 * reference / (value + reference)
    if metric == 'Spearman':
        return 100 * (1 + value) / (2 + value + reference)
    return 100 * value / (value + reference)


def main():
    baseline = json.loads((ROOT / 'docs/platform_scorecard_v364_20261104.json').read_text(encoding='utf-8'))
    scenarios = []
    for auc_delta, rank_delta in ((.005, .02), (.01, .05), (.02, .10)):
        records = []
        for row in baseline['tasks']:
            delta = auc_delta if row['task'] == 'ames' else rank_delta if row['task'] in ('clearance_microsome_az', 'half_life_obach', 'vdss_lombardo') else 0.
            value = min(1., row['raw'] + delta) if delta else row['raw']
            records.append(dict(task=row['task'], metric=row['metric'], baseline_raw=row['raw'],
                                hypothetical_raw=value, score=task_score(row['metric'], value, row['reference_score'])))
        total = sum(row['score'] for row in records) / 22
        scenarios.append(dict(ames_auc_increment=auc_delta, three_spearman_increments=rank_delta,
                              conditional_total=total, conditional_total_gain=total-baseline['task_score_mean'], tasks=records))
    value = dict(description='Conditional arithmetic only, not predicted performance, confidence intervals, or probabilities.',
                 baseline=baseline['task_score_mean'], scenarios=scenarios)
    (ROOT / 'docs/v368_conditional_score_scenarios.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{key: row[key] for key in ('ames_auc_increment', 'three_spearman_increments', 'conditional_total', 'conditional_total_gain')} for row in scenarios]))


if __name__ == '__main__':
    main()
