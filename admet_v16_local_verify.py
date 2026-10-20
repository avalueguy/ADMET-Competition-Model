"""Read-only local validation of the exact remotely verified v16 CSV."""
import csv,hashlib,json,math
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as handle:return list(csv.reader(handle))

root=Path('admet_release_v16')
proof=json.loads((root/'independent_verification.json').read_text(encoding='utf-8'))
assert proof['status']=='VERIFIED' and proof['independent_replay']
assert len(proof['task_checks'])==22 and all(x['passed'] for x in proof['task_checks'])
new=Path(proof['submission']);old=Path('admet_current_candidate.json')
old=Path(json.loads(old.read_text(encoding='utf-8'))['submission'])
assert digest(new)==proof['sha256'] and digest(old)==proof['prior_prediction_sha256'] if 'prior_prediction_sha256' in proof else digest(new)==proof['sha256']
blob=new.read_bytes();assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n')==16380
assert blob.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
a=read(new);b=read(old);c=read('admet_official/submission_template.csv')
assert len(a)==len(b)==len(c)==16380
assert a[0]==b[0]==c[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
changed=set(proof['changed']);unchanged=set(proof['unchanged']);assert len(changed|unchanged)==22 and not changed&unchanged
seen=set();actual=set()
for r,prior,template in zip(a[1:],b[1:],c[1:]):
    assert len(r)==6 and r[:4]+r[5:]==prior[:4]+prior[5:]==template[:4]+template[5:]
    key=(r[0],r[1]);assert key not in seen;seen.add(key)
    value=float(r[4]);assert math.isfinite(value)
    if r[5] in ['AUROC','AUPRC']:assert 0<=value<=1
    if r[0] in unchanged:assert r==prior
    if r[4]!=prior[4]:actual.add(r[0])
assert actual==changed and len(seen)==16379
result=dict(status='VERIFIED_LOCAL_AND_REMOTE',submission=str(new.resolve()),sha256=proof['sha256'],
    bytes=len(blob),rows=16379,tasks=22,changed=sorted(changed),unchanged=sorted(unchanged),
    baseline_sha256=digest(old),template_metadata_exact=True,encoding='UTF-8 BOM',line_endings='CRLF',
    model_replay_verified=True,local_gpu_training=False)
(root/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
