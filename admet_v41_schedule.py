"""Queue the salt-parent GPU experiment after v40 verification finishes."""
import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v41'
OUT.mkdir(parents=True, exist_ok=True)
assert os.name != 'nt' and str(ROOT) == '/home/czp123/admet_v23_20260928'
env = os.environ.copy()
env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3')
deadline = time.monotonic()+14400
while not (ROOT / 'admet_model_outputs_v40/finish_completed.json').exists():
    assert time.monotonic() < deadline
    time.sleep(15)
with (OUT / 'run.log').open('ab') as log:
    subprocess.run([str(ROOT / '.venv/bin/python'), '-u', 'admet_v41_run.py'], cwd=ROOT, env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
print('v41 trained', flush=True)
