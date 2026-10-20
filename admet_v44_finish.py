import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT/'.venv/bin/python')
report = json.loads((ROOT/'admet_model_outputs_v44/analysis.json').read_text())
assert report['status']=='EVALUATED'
if report['selected_tasks']:
    for args in [['admet_v44_run.py','final'],['admet_v44_remote_verify.py'],['admet_v44_diagnostics.py'],
                 ['admet_v44_run.py','export'],['admet_v44_collect.py']]:
        subprocess.run([PYTHON,'-u']+args,cwd=ROOT,check=True)
else:
    print('No eligible candidate; confirmed v43 retained',flush=True)
