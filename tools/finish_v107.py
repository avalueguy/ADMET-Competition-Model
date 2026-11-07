"""Wait for frozen experiments, independently verify, then assemble candidate."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(file,*args):subprocess.run([sys.executable,'-u',str(ROOT/'tools'/file),*args],cwd=ROOT,check=True)
def main():
    for _ in range(1440):
        states={n:json.loads((ROOT/f'admet_model_outputs_v{n}/pipeline_status.json').read_text()) for n in (102,103,104,106)}
        if any(s['status']=='FAILED' for s in states.values()):raise RuntimeError(states)
        if all(s['status']=='CONFIRMATION_COMPLETE' for s in states.values()):break
        time.sleep(10)
    else:raise TimeoutError('Experiments did not complete; refusing candidate.')
    for n in (102,103,104,106):
        p=ROOT/f'admet_model_outputs_v{n}/model_verification.json'
        if p.exists():assert json.loads(p.read_text())['status']=='PASS'
        else:run('v107_delivery.py','verify','--version',str(n))
    run('v107_delivery.py','assemble');run('v107_diagnostics.py')
    print(json.dumps(dict(status='ASSEMBLED_VERIFIED',version='v107',archive_pending=True)),flush=True)
if __name__=='__main__':main()
