import os
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v41'
OUT.mkdir(parents=True, exist_ok=True)
assert os.name != 'nt' and str(ROOT) == '/home/czp123/admet_v23_20260928'
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (OUT / 'features.log').open('ab') as log:
    p = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', 'admet_v41_features.py'], cwd=ROOT, env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(p.pid)
