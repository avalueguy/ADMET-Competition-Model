"""Read-only progress, no query scores or configuration changes."""
from pathlib import Path
import json
import subprocess
import time


def main():
    root=Path('/root/autodl-tmp/regression9_v379')
    result=dict(at_unix=time.time(),root=str(root))
    for name in ('run_status','launch','smoke_receipt'):
        path=root/(name+'.json')
        if path.exists():result[name]=json.loads(path.read_text())
    events=[]
    if (root/'training.log').exists():
        for line in (root/'training.log').read_text().splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
    done=[v for v in events if v.get('stage')=='curve_job_done']
    result['completed_development_curves']=done[-1]['completed'] if done else 0
    result['total_development_curves']=36
    result['model_states']=len(list((root/'outputs/curves').glob('*/*/*/state.joblib')))
    result['prediction_packages']=len(list((root/'outputs/tasks').glob('*/*/*/predictions.npz')))
    last=[]
    for path in sorted((root/'prefit_logs').glob('*.log')):
        lines=path.read_text().splitlines()
        for line in reversed(lines):
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if value.get('stage')=='neural_horizon_done':
                last.append(dict(job=path.stem,seed=value['seed'],steps=value['steps'],
                                 seconds=value['seconds'],modified=path.stat().st_mtime))
                break
    result['latest_active_training']=sorted(last,key=lambda x:x['modified'],reverse=True)[:6]
    result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    tmp=root/'progress.tmp'
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(root/'progress.json')
    print(json.dumps(result))


if __name__=='__main__':main()
