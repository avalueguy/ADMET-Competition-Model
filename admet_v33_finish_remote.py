"""After the isolated training process finishes, verify and pack its output."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v33'
assert os.name!='nt' and str(ROOT)=='/home/czp123/admet_v23_20260928'
pid=json.loads((OUT/'launch.json').read_text())['pid']
while True:
    try:os.kill(pid,0)
    except ProcessLookupError:break
    time.sleep(15)
assert (OUT/'all_completed.json').exists(),'Training ended without all_completed.json; inspect run.log'
report=json.loads((OUT/'analysis.json').read_text())
assert report['selected_tasks'],'No qualifying candidate: keep v32'
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3')
for script in ['admet_v33_remote_verify.py','admet_v33_diagnostics.py','admet_v33_collect.py']:
    with (OUT/'finish.log').open('ab') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',script],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
(OUT/'finish_completed.json').write_text(json.dumps(dict(completed=True,training_pid=pid)),encoding='utf-8')
print('Final model verification, diagnostics and lightweight archive completed',flush=True)
