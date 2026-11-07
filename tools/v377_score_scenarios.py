"""Score arithmetic from the latest card; no learning or performance forecast."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def total(card, reduction):
    return sum(100 * row['reference_score'] /
               (row['reference_score'] + row['raw'] * (1 - reduction))
               if row['metric'] == 'MAE' else row['score']
               for row in card['tasks']) / len(card['tasks'])


def main():
    card = json.loads((ROOT / 'docs/platform_scorecard_v374_20261105.json').read_text(encoding='utf-8'))
    for row in card['tasks']:
        if row['metric'] == 'MAE':
            assert abs(100 * row['reference_score'] /
                       (row['reference_score'] + row['raw']) - row['score']) < 1e-6
    cases = [dict(equal_relative_MAE_reduction_in_all_five=r,
                  conditional_total=total(card, r))
             for r in (0., .03, .05, .08, .10, .15, .20)]
    lo, hi = 0., .8
    for _ in range(70):
        mid = (lo + hi) / 2
        if total(card, mid) < 52:
            lo = mid
        else:
            hi = mid
    result = dict(source='docs/platform_scorecard_v374_20261105.json',
                  baseline=card['task_score_mean'], cases=cases,
                  equal_reduction_in_all_five_needed_for_52=hi,
                  note='Conditional arithmetic only. Equal reduction in every directly scored MAE task; other17 scores fixed. Nine-task average5percent alone does not imply this total.',
                  new_official_evaluation=False)
    (ROOT / 'docs/v377_conditional_score_scenarios.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
