"""Wait for completed frozen experiments, verify final experts, assemble release."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())
def run(file,*args):subprocess.run([sys.executable,'-u',str(ROOT/'tools'/file),*args],cwd=ROOT,check=True)

def main():
    for _ in range(720):
        status=ROOT/'admet_model_outputs_v98/pipeline_status.json'
        if status.exists():
            state=read(status)['status']
            if state=='WORKER_FAILED':raise RuntimeError(read(status))
            if state=='CONFIRMATION_COMPLETE':break
        time.sleep(10)
    else:raise TimeoutError('No completed v98 confirmation; refusing release.')
    assert read(ROOT/'admet_model_outputs_v100/model_verification.json')['status']=='PASS'
    conf=read(ROOT/'admet_model_outputs_v98/confirmation.json');assert conf['status']=='COMPLETE'
    if conf['accepted']:run('v98_delivery.py','export')
    run('v98_delivery.py','verify');run('v98_diagnostics.py');run('v101_assemble.py')
    print(json.dumps(dict(status='ASSEMBLED_VERIFIED',version='v101',archive_pending=True)),flush=True)

if __name__=='__main__':main()
