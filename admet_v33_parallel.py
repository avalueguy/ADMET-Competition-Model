"""Coordinate local remote processes without spawning agents or touching Qwen jobs."""
import os,json,subprocess,time
from pathlib import Path
import admet_v24_models as b
import admet_v33_run as run
ROOT=run.ROOT;OUT=run.OUT
assert os.name!='nt' and str(ROOT)==run.P['remote_root']
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
python=str(ROOT/'.venv/bin/python')
def launch(script,*args,log):
    with (OUT/log).open('ab') as stream:
        return subprocess.Popen([python,'-u',script,*map(str,args)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
def workers(stage):
    processes=[launch('admet_v33_worker.py',stage,j,log=f'{stage}_shard{j}.log') for j in range(3)]
    b.save(OUT/'parallel_status.json',dict(stage=stage,pids=[p.pid for p in processes],signature=run.signature()))
    while any(p.poll() is None for p in processes):time.sleep(15)
    codes=[p.returncode for p in processes];assert codes==[0,0,0],f'{stage} failed: {codes}'
workers('evaluate')
parts=[json.loads((OUT/f'analysis_shard{j}.json').read_text()) for j in range(3)]
assert all(r['status']=='EVALUATED' and r['signature']==run.signature() for r in parts)
records={r['task']:r for part in parts for r in part['tasks']};assert set(records)==set(run.P['tasks'])
ordered=[records[t] for t in run.P['tasks']]
b.save(OUT/'analysis.json',dict(signature=run.signature(),status='EVALUATED',baseline='v32',baseline_score=50.399027,
    tasks=ordered,selected_tasks=[r['task'] for r in ordered if r['selection']],official_score_prediction=None,limitations=run.P['limitations']))
assert any(r['selection'] for r in ordered),'No candidate qualified: keep v32'
# Build the shared final feature cache exactly once before concurrent fits.
p=launch('admet_v33_run.py','features',log='final_feature_prep.log')
assert p.wait()==0
p=launch('admet_v33_final_features.py',log='final_feature_prep.log');assert p.wait()==0
workers('final')
compact={}
for j in range(3):
    part=json.loads((OUT/f'test_predictions_shard{j}.json').read_text());assert part['signature']==run.signature()
    assert not set(compact)&set(part['predictions']);compact.update(part['predictions'])
b.save(OUT/'test_predictions_compact.json',compact)
for script,args in [('admet_v33_run.py',['export']),('admet_v33_remote_verify.py',[]),('admet_v33_diagnostics.py',[]),('admet_v33_collect.py',[])]:
    p=launch(script,*args,log='finish.log');assert p.wait()==0,f'{script} failed'
b.save(OUT/'finish_completed.json',dict(signature=run.signature(),completed=True,three_remote_processes=True))
b.save(OUT/'parallel_status.json',dict(stage='COMPLETE',signature=run.signature()))
print('Remote v33 evaluation, final refit, replay and collection completed',flush=True)
