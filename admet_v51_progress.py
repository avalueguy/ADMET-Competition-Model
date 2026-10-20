"""Compact experiment status without importing a training framework."""
import json
import time
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928/admet_model_outputs_v51')
pid=int((root/'run.pid').read_text())
print('RUNNING',Path(f'/proc/{pid}').exists(), 'PID',pid)
print('RECENT_MODELS',[(str(p.relative_to(root)),int(time.time()-p.stat().st_mtime))
    for p in sorted(root.glob('*/*/*.cbm'),key=lambda p:p.stat().st_mtime)[-4:]])
for name in ['analysis.json','fresh.json','all_completed.json','no_release.json']:
    p=root/name
    if p.exists():
        a=json.loads(p.read_text());print(name,a.get('status',a.get('completed')))
        for r in a.get('tasks',[]):
            s=r.get('selection')
            print(json.dumps({k:v for k,v in r.items() if k not in ['grid'] and (k!='selection' or s)},ensure_ascii=False))
print('LOG_TAIL', '\n'.join((root/'run.log').read_text()[-6500:].splitlines()[-14:]))
other=root.parent/'admet_model_outputs_v52_neural'
if (other/'run.pid').exists():
    pid=int((other/'run.pid').read_text());print('NEURAL_RUNNING',Path(f'/proc/{pid}').exists(),'PID',pid)
    print('NEURAL_LOG','\n'.join((other/'run.log').read_text()[-4500:].splitlines()[-8:]))
    if (other/'analysis.json').exists():
        a=json.loads((other/'analysis.json').read_text());print('NEURAL_STATUS',a['status'])
        for r in a.get('tasks',[]):print(json.dumps({k:v for k,v in r.items() if k!='grid'}))
other=root.parent/'admet_model_outputs_v54_graph'
if (other/'run.pid').exists():
    pid=int((other/'run.pid').read_text());print('GRAPH_RUNNING',Path(f'/proc/{pid}').exists(),'PID',pid)
    print('GRAPH_LOG','\n'.join((other/'run.log').read_text()[-4500:].splitlines()[-6:]))
    if (other/'analysis.json').exists():
        a=json.loads((other/'analysis.json').read_text());print('GRAPH_STATUS',a['status'])
        for r in a.get('tasks',[]):print(json.dumps({k:v for k,v in r.items() if k!='grid'}))
