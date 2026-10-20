import os,json,subprocess
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928');assert Path.cwd()==root;out=root/'admet_model_outputs_v64';out.mkdir(exist_ok=True);pidfile=out/'pipeline.pid'
if pidfile.exists():
    pid=int(pidfile.read_text());proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists() and b'admet_v64_pipeline.py' in proc.read_bytes():print(json.dumps(dict(status='ALREADY_RUNNING',pid=pid)));raise SystemExit()
with (out/'pipeline.log').open('ab') as log:p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','admet_v64_pipeline.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid));print(json.dumps(dict(status='STARTED',pid=p.pid)))
