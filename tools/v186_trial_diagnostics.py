"""Read-only endpoint comparison of completed frozen experimental routes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    baseline = load(ROOT / 'docs/platform_scorecard_v145_20261027.json')
    routes = []
    for number in (149, 150, 151, 153, 157, 159, 168, 172, 173, 175, 178, 181):
        folder = ROOT / f'admet_model_outputs_v{number}'
        cp = folder / 'confirmation.json'
        if not cp.exists():
            continue
        conf = load(cp)
        if conf['status'] != 'COMPLETE':
            continue
        dev = load(folder / 'development.json')
        assert dev['status'] == 'COMPLETE'
        selected = load(folder / 'frozen_selection.json')
        assert conf['signature'] == selected['signature'] == dev['signature']
        comparisons = []
        for task in dict.fromkeys(r['task'] for r in dev['records']):
            records = [r for r in dev['records'] if r['task'] == task]
            best = max(records, key=lambda r: r['objective'])
            record = next((r for r in conf['records'] if r['task'] == task), None)
            comparisons.append(dict(task=task, metric=best['metric'],
                                    official_baseline_task_score=next(r['score'] for r in baseline['tasks'] if r['task'] == task),
                                    max_minimum_development_gain=best['objective'],
                                    config=best['config'], weight=best['weight'],
                                    confirmation_gains={r['part']: r['gain'] for r in record['confirmation']} if record else None,
                                    accepted=bool(record and record['accepted']),
                                    failure_reasons=(['no_development_nomination'] if record is None else
                                                     (['nonpositive_confirmation_point'] if any(r['gain'] <= 0 for r in record['confirmation']) else []) +
                                                     ['bootstrap_uncertain_' + r['part'] for r in record['bootstrap'] if not r['lower_positive']])))
        routes.append(dict(route='v' + str(number), tasks=len(comparisons),
                           development_nominees=[r['task'] for r in selected['selected']],
                           accepted=conf['accepted'], comparisons=comparisons))
    report = dict(status='COMPLETE_REPORT_ONLY', baseline='v145', official_best=51.166,
                  completed_routes=routes, submission_decisions_changed=False,
                  model_selection_changed=False, source_data_changed=False,
                  limitations='Describes previously observed evidence. Failed configurations are not retuned or promoted using this report.')
    (ROOT / 'docs/v186_completed_trial_diagnostics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([dict(route=r['route'], tasks=r['tasks'], nominees=r['development_nominees'],
                           accepted=r['accepted']) for r in routes]))


if __name__ == '__main__':
    main()
