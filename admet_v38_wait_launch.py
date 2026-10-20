"""Schedule the complementary kernel experiment after v37 stops using the GPU."""
import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
assert os.name != 'nt' and str(ROOT) == '/home/czp123/admet_v23_20260928'
limit = time.monotonic() + 14400
while not (ROOT / 'admet_model_outputs_v37/finish_completed.json').exists():
    assert time.monotonic() < limit
    time.sleep(15)
subprocess.run([str(ROOT / '.venv/bin/python'), 'admet_v38_launch.py'], cwd=ROOT, check=True)
