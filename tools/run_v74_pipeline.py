"""Resume after selection training, then confirm, refit, verify and archive."""
from pathlib import Path
import sys,subprocess,time,json,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'admet_model_outputs_v74'

def save(value):
    p=OUT/'pipeline_status.json';p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    pid=int(sys.argv[1]);ready=OUT/'completed_fold0_fold1.json';start=time.time()
    save(dict(status='WAITING_SELECTION_TRAINING',pid=pid))
    while not ready.exists():
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Selection training stopped without completion')
        if time.time()-start>10800:raise TimeoutError('Selection stage exceeded three hours')
        time.sleep(10)
    stages=[('confirmation','tools/run_v74_confirmation.py'),('finalize','tools/finalize_v74.py'),('verify','tools/verify_v74_models.py'),('package','tools/package_v74_release.py')]
    for name,script in stages:
        save(dict(status='RUNNING',stage=name))
        with (ROOT/f'v74_{name}.log').open('w') as out:
            subprocess.run([sys.executable,'-u',script],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    save(dict(status='COMPLETE',candidate='admet_submissions/v74_v73',official_score=None))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(dict(status='FAILED',error=str(exc)));raise
