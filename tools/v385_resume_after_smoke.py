"""Resume once after a test-only correction, preserving all frozen sources."""
from pathlib import Path
import json,subprocess,sys,hashlib
ROOT=Path('/root/autodl-tmp/regression9_v385')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not(ROOT/'launch.json').exists()and not(ROOT/'outputs/development.json').exists()
 freeze=json.loads((ROOT/'source_freeze.json').read_text())
 for r in freeze['files']:assert sha(ROOT/r['name'])==r['sha256']
 amendment=dict(reason='Finite empirical rank map is not exact inverse. Amend only smoke assertion; no outer evaluation yet.',original_smoke_sha256=sha(ROOT/'v385_smoke.py'),amended_smoke_sha256=sha(ROOT/'v385_smoke_amended.py'),frozen_training_sources_unchanged=True)
 (ROOT/'smoke_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
 subprocess.run([sys.executable,'-u','v385_smoke_amended.py'],cwd=ROOT,check=True)
 with(ROOT/'training.log').open('w')as log:proc=subprocess.Popen([sys.executable,'-u','v385_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(status='STARTED',pid=proc.pid,root=str(ROOT),workers=2,smoke_amendment=amendment)
 (ROOT/'launch.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
