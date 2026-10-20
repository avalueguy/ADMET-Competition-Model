"""Wait for the isolated experiment, then independently verify and collect its artifacts."""
import json
import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v37'
assert os.name != 'nt' and str(ROOT) == '/home/czp123/admet_v23_20260928'
deadline = time.monotonic() + 14400
while not (OUT / 'all_completed.json').exists():
    assert time.monotonic() < deadline, 'Experiment exceeded four-hour audit wait'
    time.sleep(15)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3')
for script in ['admet_v37_remote_verify.py', 'admet_v37_diagnostics.py', 'admet_v37_collect.py']:
    with (OUT / 'finish.log').open('ab') as log:
        subprocess.run([str(ROOT / '.venv/bin/python'), '-u', script], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
(OUT / 'finish_completed.json').write_text(json.dumps(dict(completed=True)), encoding='utf-8')
print('v37 final replay, diagnostic and evidence completed', flush=True)
