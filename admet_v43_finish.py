"""Only finish a frozen, evaluated candidate after independent replay succeeds."""
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / '.venv/bin/python')
report = json.loads((ROOT / 'admet_model_outputs_v43/analysis.json').read_text())
assert report['status'] == 'EVALUATED'
if report['selected_tasks']:
    for args in [['admet_v43_run.py', 'final'], ['admet_v43_remote_verify.py'],
                 ['admet_v43_diagnostics.py'], ['admet_v43_run.py', 'export'], ['admet_v43_collect.py']]:
        subprocess.run([PYTHON, '-u']+args, cwd=ROOT, check=True)
else:
    print('No selected candidate; scored v42 remains recommended', flush=True)
