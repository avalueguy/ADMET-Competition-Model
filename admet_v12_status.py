"""Compact remote status, including live processes rather than stale marker alone."""
import json
from pathlib import Path
from collections import Counter


out=Path('admet_model_outputs_v12')
status=json.loads((out/'coordinator_status.json').read_text())
launch=json.loads((out/'launch.json').read_text())
live={}
for name in ['pid','finisher_pid']:
    p=Path('/proc')/str(launch[name])
    live[name]=p.exists() and p.joinpath('stat').read_text().split()[2]!='Z'
decisions=[json.loads(p.read_text()) for p in out.glob('*/decision.json')]
folds={f:len(list(out.glob('*/'+f+'_fold*.npz'))) for f in ['boost_compact','boost_fragments','kernel_adaptive']}
print(json.dumps(dict(status=status,live=live,completed_folds=folds,decisions=len(decisions),
    accepted=[r['task'] for r in decisions if r['accepted']],rejections=dict(Counter(r['reason'] for r in decisions if not r['accepted'])))))
for p in sorted(Path('logs').glob('v12*.log')):
    if p.name=='v12_tests.log':continue
    lines=p.read_text(errors='replace').splitlines()
    useful=[s for s in lines if s.startswith('{') or 'Error' in s or 'Traceback' in s]
    if useful:print(p.name, useful[-1])
