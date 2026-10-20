import os
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
assert os.name != 'nt' and str(ROOT) == '/home/czp123/admet_v23_20260928'
with (ROOT / 'admet_model_outputs_v37/coordinator.log').open('ab') as log:
    p = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', 'admet_v37_finish.py'], cwd=ROOT,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(p.pid)
