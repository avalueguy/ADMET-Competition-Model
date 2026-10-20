import os,subprocess,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v33'
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
processes=[]
for j in range(3):
    with (OUT/f'final_shard{j}.log').open('ab') as log:
        processes.append(subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','admet_v33_worker.py','final',str(j)],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT))
(OUT/'parallel_status.json').write_text(json.dumps(dict(stage='final',pids=[p.pid for p in processes])),encoding='utf-8')
while any(p.poll() is None for p in processes):time.sleep(15)
assert [p.returncode for p in processes]==[0,0,0]
compact={}
for j in range(3):
    part=json.loads((OUT/f'test_predictions_shard{j}.json').read_text());assert not set(compact)&set(part['predictions']);compact.update(part['predictions'])
(OUT/'test_predictions_compact.json').write_text(json.dumps(compact),encoding='utf-8')
for script,args in [('admet_v33_run.py',['export']),('admet_v33_remote_verify.py',[]),('admet_v33_diagnostics.py',[]),('admet_v33_collect.py',[])]:
    with (OUT/'finish.log').open('ab') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',script,*args],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
(OUT/'finish_completed.json').write_text(json.dumps(dict(completed=True)),encoding='utf-8')
(OUT/'parallel_status.json').write_text(json.dumps(dict(stage='COMPLETE')),encoding='utf-8')
