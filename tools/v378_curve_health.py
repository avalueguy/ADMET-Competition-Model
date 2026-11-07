"""Read training-only checkpoints and resource usage, without query evaluation."""
from pathlib import Path
import json
import subprocess

ROOT=Path('/root/autodl-tmp/regression9_v378')


def main():
    records=[]
    for path in [ROOT/'training.log']+sorted((ROOT/'prefit_logs').glob('*.log')):
        events=[]
        for line in path.read_text(errors='replace').splitlines():
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if value.get('stage')=='neural_horizon_done':events.append(value)
        if events:
            value=events[-1]
            records.append(dict(task=value['task'],part=value['part'],base=value['base'],
                                seed=value['seed'],steps=value['steps'],seconds=value['seconds'],
                                train_loss_last100=value['train_loss_last100']))
    prefit=ROOT/'parallel_prefit_results.json'
    progress=json.loads(prefit.read_text()) if prefit.exists() else dict(results=[],total=34)
    value=dict(training_only=True,query_scores_read=False,
               completed_parallel_curves=len(progress['results']),
               failed_parallel_curves=sum(x['exit_code']!=0 for x in progress['results']),
               states=len(list((ROOT/'outputs/curves').glob('*/*/*/state.joblib'))),
               horizons_logged=sum(1 for r in records if r['steps']==4800),latest_by_job=records,
               gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
    (ROOT/'training_health.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
