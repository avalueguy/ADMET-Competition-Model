"""Compact read-only report of the three frozen specialist experiments."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for version in (115, 116, 117):
    folder = root / f'admet_model_outputs_v{version}'
    result = dict(version=version)
    for stage in ('development', 'frozen_selection', 'confirmation'):
        path = folder / (stage + '.json')
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        info = dict(status=doc['status'])
        if stage == 'development':
            best = {}
            for row in doc['records']:
                objective = min(p['gain'] for p in row['parts'])
                if row['task'] not in best or objective > best[row['task']]['objective']:
                    best[row['task']] = dict(objective=objective,
                        gains=[p['gain'] for p in row['parts']])
            info['tasks'] = best
        elif stage == 'frozen_selection':
            info['selected'] = [r['task'] for r in doc['selected']]
        else:
            info['accepted'] = doc.get('accepted')
            info['records'] = [dict(task=r['task'], accepted=r['accepted'],
                gains=[p['gain'] for p in r['confirmation']]) for r in doc['records']]
        result[stage] = info
    print(json.dumps(result), flush=True)
