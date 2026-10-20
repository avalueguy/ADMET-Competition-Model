"""Durable bounded completion: wait for authorized experiments, replay, export once."""
import os,time,json,subprocess
from pathlib import Path
ROOT=Path('/home/czp123/admet_v23_20260928');assert os.name!='nt' and Path.cwd()==ROOT
OUT=ROOT/'admet_model_outputs_v64';OUT.mkdir(exist_ok=True)
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
start=time.time();done=set()
def save(value):
    temp=OUT/'pipeline_status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(OUT/'pipeline_status.json')
while len(done)<4:
    if time.time()-start>14400:raise TimeoutError('Four-hour experiment completion bound exceeded')
    for version in [61,62,63,65]:
        if version in done:continue
        src=ROOT/f'admet_model_outputs_v{version}'
        if not (src/'all_completed.json').exists():continue
        if version==61 and not (src/'independent_chemical_gate.json').exists():continue
        if (src/'release.json').exists() and not (src/'model_replay.json').exists():
            print(json.dumps(dict(stage='REPLAY_START',version=version)),flush=True)
            subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/f'admet_v{version}_verify.py')],cwd=ROOT,env=env,check=True)
        elif not (src/'release.json').exists():assert (src/'no_release.json').exists(),version
        done.add(version);save(dict(status='WAITING_FOR_SOURCES',completed_sources=sorted(done),elapsed_seconds=int(time.time()-start)))
    if len(done)<4:time.sleep(10)
subprocess.run([str(ROOT/'.venv/bin/python'),'-u','admet_v64_finish.py'],cwd=ROOT,env=env,check=True)
save(dict(status='COMPLETE',completed_sources=sorted(done),elapsed_seconds=int(time.time()-start),automatic_upload=False))
print(json.dumps(dict(status='COMPLETE')),flush=True)
