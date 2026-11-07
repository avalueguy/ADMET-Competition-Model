"""Start the frozen remote coefficient experiment after preflight checks."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path('/root/autodl-tmp/regression9_v389')
def main():
    assert not(ROOT/'launch.json').exists()
    plan=json.loads((ROOT/'plan.json').read_text());assert hashlib.sha256((ROOT/'v389_combination.py').read_bytes()).hexdigest()==plan['source_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for v in('v386','v387'):
        exp=ROOT.parent/f'regression9_{v}';assert json.loads((exp/'run_status.json').read_text())['stage']=='EVIDENCE_READY'
        for r in json.loads((exp/'source_freeze.json').read_text())['files']:
            assert hashlib.sha256((exp/r['name']).read_bytes()).hexdigest()==r['sha256']
    with(ROOT/'training.log').open('w')as log:
        p=subprocess.Popen([sys.executable,'-u','v389_combination.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(status='STARTED',pid=p.pid)))
if __name__=='__main__':main()
