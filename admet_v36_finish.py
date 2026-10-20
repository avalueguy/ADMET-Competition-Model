import os,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v36';OUT.mkdir(parents=True,exist_ok=True)
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
while not all(p.exists() for p in [ROOT/'admet_model_outputs_v33/finish_completed.json',ROOT/'admet_model_outputs_v34/completed.json',ROOT/'admet_model_outputs_v35/completed.json']):time.sleep(15)
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
for script in ['admet_v36_run.py','admet_v36_remote_verify.py','admet_v36_collect.py']:
    with (OUT/'run.log').open('ab') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',script],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
(OUT/'completed.json').write_text(json.dumps(dict(completed=True)),encoding='utf-8')
print('Final combined release completed',flush=True)
