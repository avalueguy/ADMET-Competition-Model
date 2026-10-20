import json
import os
import subprocess
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928');assert os.name!='nt' and Path.cwd()==root
out=root/'admet_model_outputs_v54_graph';out.mkdir(exist_ok=True);pidfile=out/'run.pid'
if pidfile.exists():
    pid=int(pidfile.read_text());p=Path(f'/proc/{pid}/cmdline')
    if p.exists() and b'admet_v54_graph.py' in p.read_bytes():
        print(json.dumps(dict(status='ALREADY_RUNNING',pid=pid)));raise SystemExit(0)
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
with (out/'run.log').open('ab') as log:
    p=subprocess.Popen([str(root/'.venv/bin/python'),'-u',str(root/'admet_v54_graph.py')],cwd=root,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid));print(json.dumps(dict(status='STARTED',pid=p.pid)))
