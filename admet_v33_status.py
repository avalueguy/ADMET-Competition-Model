"""Compact factual progress; no score projections."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v33'
if (OUT/'parallel_status.json').exists():print((OUT/'parallel_status.json').read_text())
for shard in range(3):
    path=OUT/f'analysis_shard{shard}.json'
    if path.exists():
        obj=json.loads(path.read_text());print(json.dumps(dict(shard=shard,status=obj['status'],
            evaluated=len(obj['tasks']),selected=[dict(task=r['task'],valid_gain=r['selection']['valid_gain'],
                cv_mean=r['selection']['cv_mean'],weight=r['selection']['weight']) for r in obj['tasks'] if r['selection']]),ensure_ascii=False))
for log in ['evaluate_shard0.log','evaluate_shard1.log','evaluate_shard2.log','finish.log','parallel.log']:
    p=OUT/log
    if p.exists():
        lines=[x for x in p.read_text(errors='replace').splitlines() if x.startswith('{') or any(t in x for t in ['Error','Traceback','Assertion'])]
        print(log, '\n'.join(lines[-2:]))
print('completed', (OUT/'finish_completed.json').exists())
