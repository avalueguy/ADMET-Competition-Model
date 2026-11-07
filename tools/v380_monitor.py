"""Compact training progress without evaluating query labels."""
from pathlib import Path
import json
import subprocess
import time

ROOT=Path('/root/autodl-tmp/regression9_v380')


def main():
    result=dict(at_unix=time.time())
    for name in ('run_status','launch','smoke_receipt'):
        path=ROOT/(name+'.json')
        if path.exists():result[name]=json.loads(path.read_text())
    states=list((ROOT/'outputs/tasks').glob('*/*/*/predictions.npz'))
    result['completed_development_fits']=sum(p.parts[-3] in ('fold0','fold1') for p in states)
    result['total_development_fits']=104
    result['prediction_packages']=len(states)
    if (ROOT/'training.log').exists():
        events=[]
        for line in (ROOT/'training.log').read_text(errors='replace').splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
        if events:
            result['latest_event']={k:v for k,v in events[-1].items() if k in ('stage','task','part','config','completed','total','seconds')}
    result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    (ROOT/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
