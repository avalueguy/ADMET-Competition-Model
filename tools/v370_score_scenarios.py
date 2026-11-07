"""Conditional MAE scoring arithmetic, explicitly not a performance forecast."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def total(card, reduction):
    return sum(100 * r['reference_score'] / (r['reference_score'] + r['raw'] * (1-reduction))
               if r['metric'] == 'MAE' else r['score'] for r in card['tasks']) / 22


def main():
    card = json.loads((ROOT / 'docs/platform_scorecard_v364_20261104.json').read_text(encoding='utf-8'))
    cases = [dict(five_MAE_relative_reduction=r, conditional_total=total(card, r)) for r in (0., .03, .05, .08, .10, .15, .20)]
    lo, hi = 0., .8
    for _ in range(70):
        mid = (lo + hi) / 2
        if total(card, mid) < 52:
            lo = mid
        else:
            hi = mid
    value = dict(description='Conditional arithmetic, not a forecast. Same reduction in all five MAE tasks; all other task scores fixed.',
        baseline=card['task_score_mean'], scenarios=cases, required_equal_MAE_reduction_for_52=hi,
        no_new_official_evaluation=True)
    (ROOT / 'docs/v370_conditional_score_scenarios.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    progress = json.loads((ROOT / 'docs/v370_stage5_progress.json').read_text(encoding='utf-8'))
    launch = json.loads((ROOT / 'experiments/transformer_stage5_v370/launch.json').read_text())
    smoke = json.loads((ROOT / 'experiments/transformer_stage5_v370/outputs/smoke.json').read_text())
    progress.update(status='TRAINING', remote_pid=launch['pid'], signature=smoke['signature'], smoke='PASS', parameters=smoke['parameters'])
    (ROOT / 'docs/v370_stage5_progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
