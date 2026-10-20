import os,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parent
assert os.name!='nt' and str(root)=='/home/czp123/admet_v23_20260928'
out=root/'admet_model_outputs_v31';out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (out/'run.log').open('ab') as log:
    p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','admet_v31_run.py','all'],cwd=root,env=env,
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid),indent=2),encoding='utf-8');print(p.pid)
