"""Wait for only this task's model artifacts, then merge and collect evidence."""
import json,os,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
out=ROOT/'admet_model_outputs_v29';out.mkdir(parents=True,exist_ok=True)
start=time.time()
while True:
    required=[ROOT/f'admet_model_outputs_v{v}/test_predictions_compact.json' for v in [27,28,31]]
    if all(p.exists() for p in required):break
    if time.time()-start>14400:raise RuntimeError('Model artifacts unfinished after bounded coordination window')
    time.sleep(15)
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
subprocess.run([str(ROOT/'.venv/bin/python'),'-u','admet_v29_run.py','all'],cwd=ROOT,env=env,check=True)
subprocess.run([str(ROOT/'.venv/bin/python'),'admet_v29_collect.py'],cwd=ROOT,env=env,check=True)
(out/'completed.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-start),indent=2),encoding='utf-8')
