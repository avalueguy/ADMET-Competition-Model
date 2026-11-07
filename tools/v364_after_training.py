"""Run verification and collect artifacts once the existing remote training finishes."""
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 while True:
  status=json.loads((ROOT/'run_status.json').read_text())
  if status['stage']=='FAILED':raise RuntimeError(status.get('traceback','Training failed'))
  if status['stage']=='TRAINING_COMPLETE':break
  time.sleep(20)
 for name in ('v364_verify_remote.py','v364_collect_remote.py'):
  subprocess.run([sys.executable,'-u',str(ROOT/name)],cwd=ROOT,check=True)
 print(json.dumps(dict(stage='EVIDENCE_READY')),flush=True)
if __name__=='__main__':main()
