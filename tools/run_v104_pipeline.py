"""Structure generation then frozen development and confirmation."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'admet_model_outputs_v104'
def main():
    OUT.mkdir(exist_ok=True)
    for file,args,stage in [('v104_sparse_margin.py',['protocol'],'PROTOCOL'),('v104_sparse_features.py',[],'FEATURES')]+[('v104_sparse_margin.py',[s],s.upper()) for s in ('smoke','develop','confirm')]:
        (OUT/'pipeline_status.json').write_text(json.dumps(dict(status=stage)))
        code=subprocess.call([sys.executable,'-u',str(ROOT/'tools'/file),*args],cwd=ROOT)
        if code:
            (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='FAILED',stage=stage,returncode=code)));raise SystemExit(code)
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(status='CONFIRMATION_COMPLETE')))
if __name__=='__main__':main()
