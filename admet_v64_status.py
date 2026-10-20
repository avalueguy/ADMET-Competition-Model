"""Read-only compact progress for the pending completion round."""
import json,os
from pathlib import Path

root=Path('/home/czp123/admet_v23_20260928')
assert os.name!='nt' and Path.cwd()==root
for version in (61,62,63,65,64):
    dest=root/f'admet_model_outputs_v{version}'
    report={'version':version}
    for filename in ('analysis.json','fresh.json','independent_chemical_gate.json',
                     'model_replay.json','release.json','all_completed.json',
                     'pipeline_status.json'):
        file=dest/filename
        if file.exists():
            value=json.loads(file.read_text())
            tasks=value.get('tasks',[])
            report[filename]={key:value[key] for key in ('status','sha256','elapsed_seconds') if key in value}
            if isinstance(tasks,list):
                report[filename].update(count=len(tasks),selected=[t['task'] for t in tasks if t.get('selection')],passed=[t['task'] for t in tasks if t.get('pass_gate')])
    for filename in ('run.log','pipeline.log','fold1_worker.log','fold2_worker.log','valid_worker.log'):
        file=dest/filename
        if file.exists():
            lines=file.read_text(errors='replace').splitlines()
            report[filename]=lines[-2:]
    print(json.dumps(report),flush=True)
