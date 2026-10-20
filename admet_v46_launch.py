import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
OUT=ROOT/'admet_model_outputs_v46'
OUT.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
with (OUT/'run.log').open('ab') as log:
    p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'admet_v46_run.py'),'all'],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(OUT/'pid.txt').write_text(str(p.pid))
print(p.pid)
