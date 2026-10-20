"""Finish active fold, then distribute remaining disjoint partitions and resume selection."""
import os,time,json,signal,subprocess
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928');assert Path.cwd()==root and os.name!='nt';out=root/'admet_model_outputs_v62';pid=int((out/'run.pid').read_text());boundary=out/'cv/fold0/621018/predictions.npz'
while not boundary.exists():
    proc=Path(f'/proc/{pid}/cmdline')
    if not proc.exists():raise RuntimeError('Source process exited before completed boundary')
    time.sleep(3)
proc=Path(f'/proc/{pid}/cmdline')
if proc.exists() and b'admet_v62_run.py' in proc.read_bytes():
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not proc.exists():break
        time.sleep(1)
    assert not proc.exists(),'Original training did not stop cleanly'
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
workers=[]
for partition in ['fold1','fold2','valid']:
    log=(out/(partition+'_worker.log')).open('ab');p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','admet_v62_worker.py',partition],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);workers.append((partition,p,log));print(json.dumps(dict(stage='PARTITION_STARTED',partition=partition,pid=p.pid)),flush=True)
(out/'parallel_workers.json').write_text(json.dumps({name:p.pid for name,p,log in workers},indent=2))
codes={}
for name,p,log in workers:codes[name]=p.wait();log.close()
assert all(code==0 for code in codes.values()),codes
with (out/'run.log').open('ab') as log:
    p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','admet_v62_run.py','all'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'run.pid').write_text(str(p.pid));print(json.dumps(dict(stage='SELECTION_RESUMED',pid=p.pid)),flush=True)
