"""Reconcile observed normalized scores and quantify task headroom; no model tuning."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalized(metric, raw, reference):
    if metric == 'MAE':
        return 100 * reference / (raw + reference)
    if metric == 'Spearman':
        return 100 * (raw + 1) / (raw + reference + 2)
    return 100 * raw / (raw + reference)


def main():
    history = json.loads((ROOT / 'docs/platform_history_edge_audit_20261027.json').read_text(encoding='utf-8'))
    records = history['records']
    checks = []
    for record in records:
        for task, item in record['tasks'].items():
            expected = normalized(item['metric'], item['raw_metric'], item['reference_score'])
            error = abs(expected - item['score'])
            assert error <= .00000051, (record['submitted_at'], task, expected, item)
            checks.append(error)
        assert abs(sum(item['score'] for item in record['tasks'].values()) / 22 - record['score']) < .000501
    baseline = next(r for r in records if r['submitted_at'] == '2026/10/27 17:22')
    tasks = []
    for task, item in baseline['tasks'].items():
        metric, raw, reference = item['metric'], item['raw_metric'], item['reference_score']
        best = normalized(metric, 0 if metric == 'MAE' else 1, reference)
        # Scenario sensitivities only; these are not predicted model improvements.
        scenario_raw = raw * .9 if metric == 'MAE' else min(1, raw + .1) if metric == 'Spearman' else min(1, raw + .01)
        points = normalized(metric, scenario_raw, reference) - normalized(metric, raw, reference)
        tasks.append(dict(task=task, metric=metric, raw_metric=raw, current_score=item['score'],
                          empirical_score_at_mathematical_best=best,
                          normalized_headroom=best - item['score'],
                          scenario=('MAE reduced10%' if metric == 'MAE' else 'Spearman increased0.1' if metric == 'Spearman' else 'AUROC/AUPRC increased0.01'),
                          scenario_raw_metric=scenario_raw, scenario_task_points=points,
                          scenario_total_points=points / 22))
    mae = [r for r in tasks if r['metric'] == 'MAE']
    rank = [r for r in tasks if r['metric'] == 'Spearman']
    report = dict(status='PASS', data_source='Observed official scorecards, not hidden labels',
                  observed_records=len(records), normalized_items_checked=len(checks),
                  maximum_score_rounding_residual=max(checks),
                  all_total_means_reconciled_with_rounding=True,
                  formula_status='Empirical identity reproduced over every supplied official record; not a fetched published scoring specification',
                  formulas={'MAE': '100*reference/(raw+reference)',
                            'Spearman': '100*(raw+1)/(raw+reference+2)',
                            'AUROC/AUPRC': '100*raw/(raw+reference)'},
                  baseline='v145', baseline_official_score=51.166,
                  required_sum_task_point_gain_for_52=52 * 22 - sum(r['current_score'] for r in tasks),
                  hypothetical_all_MAE_10percent_reduction_total_gain=sum(r['scenario_total_points'] for r in mae),
                  hypothetical_all_Spearman_plus01_total_gain=sum(r['scenario_total_points'] for r in rank),
                  tasks=tasks, model_selection_changed=False, acceptance_rules_changed=False,
                  limitations='Arithmetic sensitivities, not forecasts. Reused public labels and current model limitations remain; no test labels recovered or inferred.')
    (ROOT / 'docs/v184_official_objective_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('tasks', 'formulas', 'limitations')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
