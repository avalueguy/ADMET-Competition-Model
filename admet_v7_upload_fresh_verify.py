"""Validate a fresh official-header CSV and record failed headerless workaround."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

auditpath=Path('admet_diagnostics/v7_upload_fresh.json')
audit=json.loads(auditpath.read_text(encoding='utf8'))
source=Path(audit['source']);output=Path(audit['output'])
def rows(path):return list(csv.reader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
actual=rows(output);original=rows(source);template=rows(Path('admet_official/submission_template.csv'))
assert actual==original and len(actual)==16380
assert actual[0]==template[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
assert all(len(r)==6 for r in actual)
assert [i+1 for i,r in enumerate(actual) if r[0]=='task']==[1]
assert all(r[:4]+r[5:]==t[:4]+t[5:] for r,t in zip(actual,template))
assert all(math.isfinite(float(r[4])) for r in actual[1:])
assert all(0<=float(r[4])<=1 for r in actual[1:] if r[5] in ['AUROC','AUPRC'])
blob=output.read_bytes();assert len(blob.splitlines())==16380 and b'\r' not in blob
assert not blob.startswith(b'\xef\xbb\xbf') and not blob.endswith(b'\n')
assert hashlib.sha256(blob).hexdigest()==audit['sha256']
assert hashlib.md5(blob).hexdigest()==audit['md5']
prior=Path('admet_diagnostics/v7_upload_format_fix.json')
previous=json.loads(prior.read_text(encoding='utf8'));previous['platform_acceptance']='user reports rejected: line 16381: unknown task: task'
prior.write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf8')
assert hashlib.md5(blob).digest()!=hashlib.md5(Path(previous['output']).read_bytes()).digest()
audit.update(independent_verification=True,all_field_strings_identical_to_original=True,task_count=22,bytes=len(blob),
             physical_lines=16380,header_positions=[1],blank_records=0)
auditpath.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
statepath=Path('admet_v7_remote_run.json');state=json.loads(statepath.read_text(encoding='utf8'))
state.update(upload_compatibility_platform_acceptance=previous['platform_acceptance'],recommended_upload=str(output),
             recommended_upload_sha256=audit['sha256'],recommended_upload_platform_acceptance='not yet tested')
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:audit[k] for k in ['sha256','md5','bytes','physical_lines','data_rows','header_positions','independent_verification','all_field_strings_identical_to_original']}))
