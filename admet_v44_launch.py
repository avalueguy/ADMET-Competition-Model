import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'admet_model_outputs_v44'
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
OUT.mkdir(parents=True,exist_ok=True)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
with (OUT/'run.log').open('ab') as log:
    p = subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','admet_v44_run.py',sys.argv[1] if len(sys.argv)>1 else 'screen'],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
