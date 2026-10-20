import os,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v33'
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
for script in ['admet_v33_remote_verify.py','admet_v33_diagnostics.py','admet_v33_collect.py']:
    with (OUT/'finish.log').open('ab') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',script],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
(OUT/'finish_completed.json').write_text(json.dumps(dict(completed=True)),encoding='utf-8')
(OUT/'parallel_status.json').write_text(json.dumps(dict(stage='COMPLETE')),encoding='utf-8')
