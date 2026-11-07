"""Run fixed sparse boosting protocol without early-stopping query labels."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'admet_model_outputs_v106'
def main():
    OUT.mkdir(exist_ok=True)
    for s in ('smoke','develop','confirm'):
        (OUT/'pipeline_status.json').write_text(json.dumps(dict(status=s.upper())))
        code=subprocess.call([sys.executable,'-u',str(ROOT/'tools/v106_sparse_boost.py'),s],cwd=ROOT)
        if code:
            (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='FAILED',stage=s,returncode=code)));raise SystemExit(code)
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='CONFIRMATION_COMPLETE')))
if __name__=='__main__':main()
