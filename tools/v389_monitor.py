from pathlib import Path
import json,subprocess
ROOT=Path('/root/autodl-tmp/regression9_v389')
value=dict(stage=json.loads((ROOT/'run_status.json').read_text())if(ROOT/'run_status.json').exists()else'NOT_STARTED',
    component_packages=len(list((ROOT/'components').glob('*/*/*.npz'))),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
if(ROOT/'confirmation.json').exists():value['confirmed']=[dict(task=r['task'],a=r['a'],b=r['b'],eligible=r['experimental_eligible'],parts=r['confirmation'])for r in json.loads((ROOT/'confirmation.json').read_text())['records']]
print(json.dumps(value))
