"""Compact progress without interpreting unfinished candidates as improvements."""
import json
import sys
from collections import Counter
from pathlib import Path
from admet_v7_runner import protocol,OUT

p=protocol();tasks=sorted(x.name for x in Path('admet_competition_data/data_benchmark').iterdir() if x.is_dir())
completed={f:len(list(OUT.glob('*/'+f+'_cv.json'))) for f in p['families_all']+['chemberta_ft']}
decisions=[]
for task in tasks:
    file=OUT/task/'decision.json'
    if file.exists():
        d=json.loads(file.read_text());decisions.append(dict(task=task,accepted=d['accepted'],family=d['selection']['family'],reason=d['reason'],before=d.get('before'),after=d.get('after')))
if '--compact' in sys.argv:
    print(json.dumps(dict(completed_models=completed,decided=len(decisions),accepted=[x for x in decisions if x['accepted']],reasons=dict(Counter(x['reason'] for x in decisions))),indent=2))
else:print(json.dumps(dict(completed_models=completed,decisions=decisions),indent=2))
