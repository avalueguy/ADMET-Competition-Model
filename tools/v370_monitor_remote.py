"""Read compact progress without mutating the experiment or exposing credentials."""
from pathlib import Path
import json
import time

ROOT = Path('/root/autodl-tmp/transformer_stage5_v370')


def main():
    status = json.loads((ROOT / 'run_status.json').read_text()) if (ROOT / 'run_status.json').exists() else {}
    log = (ROOT / 'training.log').read_text() if (ROOT / 'training.log').exists() else ''
    records = []
    for line in log.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    start = records[0] if records else {}
    done = [r for r in records if r.get('stage') == 'job_done']
    predictions = list((ROOT / 'outputs/tasks').rglob('predictions.npz'))
    result = dict(status=status, bootstrap=json.loads((ROOT / 'bootstrap_status.json').read_text()),
        last_records=records[-3:], completed_prediction_jobs=len(predictions), last_completed_job=done[-1] if done else None,
        seconds_since_status=round(time.time()-status.get('at_unix', time.time()),1),
        traceback_seen='Traceback (most recent call last)' in log)
    for name in ('development', 'confirmation'):
        path = ROOT / 'outputs' / (name + '.json')
        if path.exists():
            value = json.loads(path.read_text())
            result[name] = [dict(task=r['task'], mode=r.get('mode', r.get('best', {}).get('mode')),
                weight=r.get('weight', r.get('best', {}).get('weight')),
                gains={p['part']:p['gain'] for p in r.get('confirmation', r.get('best', {}).get('parts', []))},
                strict=r.get('strict_accepted'), experimental=r.get('experimental_eligible')) for r in value['records']]
    print(json.dumps(result))


if __name__ == '__main__':
    main()
