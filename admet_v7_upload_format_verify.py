"""Independent standard-library validation of the headerless upload workaround."""
import csv
import io
import json
import math
import hashlib
from pathlib import Path

audit_path=Path('admet_diagnostics/v7_upload_format_fix.json')
audit=json.loads(audit_path.read_text(encoding='utf8'))
source=Path(audit['source']);output=Path(audit['output'])
def read(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'))))
original=read(source);fixed=read(output);template=read(Path('admet_official/submission_template.csv'))
assert original[0]==template[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
assert fixed==original[1:] # Exact field strings, including every decimal and empty ID.
assert len(fixed)==16379 and len({r[0] for r in fixed})==22
assert all(len(r)==6 for r in fixed)
assert all(r[:4]+r[5:]==t[:4]+t[5:] for r,t in zip(fixed,template[1:]))
assert all(math.isfinite(float(r[4])) for r in fixed)
assert all(0<=float(r[4])<=1 for r in fixed if r[5] in ['AUROC','AUPRC'])
assert not any(r[0]=='task' for r in fixed)
blob=output.read_bytes()
assert b'\xef\xbb\xbf' not in blob and b'\r' not in blob
assert len(blob.splitlines())==16379
assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['source_sha256']
assert hashlib.sha256(blob).hexdigest()==audit['sha256']
audit.update(independent_verification=True,task_count=22,blank_drug_ids=sum(r[2]=='' for r in fixed))
audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
state_path=Path('admet_v7_remote_run.json');state=json.loads(state_path.read_text(encoding='utf8'))
state['upload_compatibility_file']=str(output)
state['upload_compatibility_sha256']=audit['sha256']
state['upload_compatibility_format']='headerless UTF-8 without BOM, LF; same exact predictions'
state['upload_compatibility_platform_acceptance']='not yet tested'
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(rows=16379,tasks=22,columns=6,headers=0,prediction_strings_unchanged=True,sha256=audit['sha256']),indent=2))
