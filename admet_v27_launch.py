"""Detach only the versioned ADMET process in the authorized remote workspace."""
import os,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
out=ROOT/'admet_model_outputs_v27';out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
with (out/'run.log').open('ab') as log:
    process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','admet_v27_run.py','all'],cwd=ROOT,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
info=dict(pid=process.pid,stage='all',log=str(out/'run.log'),local_training=False)
(out/'launch.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info))
