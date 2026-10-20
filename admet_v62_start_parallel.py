import subprocess,json
from pathlib import Path
root=Path('/home/czp123/admet_v23_20260928');assert Path.cwd()==root;out=root/'admet_model_outputs_v62'
with (out/'parallel.log').open('ab') as log:p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','admet_v62_parallel.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'parallel.pid').write_text(str(p.pid));print(json.dumps(dict(status='STARTED',dispatcher=p.pid)))
