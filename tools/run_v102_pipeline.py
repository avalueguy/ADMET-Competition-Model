"""Sequential frozen development/confirmation, with explicit failure status."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'admet_model_outputs_v102'
def main():
    OUT.mkdir(exist_ok=True)
    for stage in ('smoke','develop','confirm'):
        (OUT/'pipeline_status.json').write_text(json.dumps(dict(status=stage.upper())))
        code=subprocess.call([sys.executable,'-u',str(ROOT/'tools/v102_count_margin.py'),stage],cwd=ROOT)
        if code:
            (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='FAILED',stage=stage,returncode=code)));raise SystemExit(code)
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='CONFIRMATION_COMPLETE')))
if __name__=='__main__':main()
