"""Finish this training batch once all frozen experiment routes complete."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSIONS=(134,135,136,137,139,141,143)
def main():
    last=None
    for _ in range(360):
        states={}
        for v in VERSIONS:
            path=ROOT/f'admet_model_outputs_v{v}/confirmation.json'
            states[v]=json.loads(path.read_text())['status'] if path.exists() else 'TRAINING_DEVELOPMENT'
        if states!=last:print(json.dumps(states),flush=True);last=states
        if all(x=='COMPLETE' for x in states.values()):break
        time.sleep(30)
    else:raise RuntimeError('Training batch did not complete within three hours; preserve all artifacts.')
    for script,log in [('v144_verify_and_assemble.py','v144_verify.log'),('v144_package.py','v144_package.log')]:
        with (ROOT/log).open('w') as output:subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE_READY_FOR_LOCAL_DELIVERY',flush=True)
if __name__=='__main__':main()
