import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v42'
OUT.mkdir(parents=True, exist_ok=True)
assert os.name != 'nt' and str(ROOT) == '/home/czp123/admet_v23_20260928'
deadline = time.monotonic()+14400
while not (ROOT / 'admet_model_outputs_v41/completed.json').exists():
    assert time.monotonic() < deadline
    time.sleep(15)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3')
for script in ['admet_v41_remote_verify.py', 'admet_v42_run.py', 'admet_v42_diagnostics.py', 'admet_v42_collect.py']:
    with (OUT / 'run.log').open('ab') as log:
        subprocess.run([str(ROOT / '.venv/bin/python'), '-u', script], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
(OUT / 'completed.json').write_text('{"completed": true}', encoding='utf-8')
print('v42 verified candidate collected', flush=True)
