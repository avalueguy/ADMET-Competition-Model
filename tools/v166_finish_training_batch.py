"""Wait for six immutable routes, then verify and preserve the batch."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ROUTES=(149,150,151,153,157,159)
OUT=ROOT/'admet_model_outputs_v161'

def save(value):
    OUT.mkdir(exist_ok=True)
    (OUT/'pipeline_status.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    last=None
    for _ in range(720):
        complete=[]
        for n in ROUTES:
            p=ROOT/f'admet_model_outputs_v{n}/confirmation.json'
            if p.exists() and json.loads(p.read_text()).get('status')=='COMPLETE':complete.append(n)
        if complete!=last:
            status=dict(status='WAITING_TRAINING',completed_routes=complete,remaining_routes=[n for n in ROUTES if n not in complete])
            save(status);print(json.dumps(status),flush=True);last=complete
        if len(complete)==len(ROUTES):break
        time.sleep(30)
    else:raise TimeoutError('Training did not complete within 6h. Preserve partial results.')
    for script,log in [('v161_verify_and_assemble.py','v161_verify.log'),('v161_package.py','v161_package.log')]:
        save(dict(status='RUNNING',stage=script,completed_routes=complete))
        with (ROOT/log).open('a',encoding='utf-8') as f:
            subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    save(dict(status='COMPLETE',submission_created=(OUT/'evidence.json').exists(),no_confirmed_candidate=(OUT/'no_candidate.json').exists(),browser_submission_not_performed=True))
    print('COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        save(dict(status='FAILED',traceback=traceback.format_exc()));raise
