"""Independently replay v24 predictions and the official CSV contract locally.

Only CSV/JSON predictions and metadata are required. No tensor framework,
features or model checkpoints are loaded on the user's machine.
"""
import csv,io,json,hashlib,math
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def ranks(x):
    order=sorted(range(len(x)),key=lambda i:x[i]);result=[0.]*len(x);s=0
    while s<len(order):
        e=s+1
        while e<len(order) and x[order[e]]==x[order[s]]:e+=1
        for i in order[s:e]:result[i]=(s+e)/2/len(x)
        s=e
    return result
def main():
    dest=ROOT/'admet_submissions/v24';new=next(dest.glob('*.csv'));old=ROOT/'admet_submissions/v23/v23_upload.csv'
    proof=json.loads((dest/'verification.json').read_text())
    analysis=json.loads((ROOT/'admet_model_outputs_v24/analysis.json').read_text())
    assert sha(new)==proof['sha256'];assert sha(old)==proof['baseline_sha256']
    a,b,t=readrows(new),readrows(old),readrows(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(b)==len(t)==16380 and a[0]==b[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    data=new.read_bytes();assert data.startswith(b'\xef\xbb\xbf');assert data.count(b'\r\n')==data.count(b'\n')==16380
    changed=Counter();seen=set();bytask=defaultdict(list)
    for current,prior,template in zip(a[1:],b[1:],t[1:]):
        assert len(current)==6 and current[:4]+current[5:]==prior[:4]+prior[5:]==template[:4]+template[5:]
        key=(current[0],current[1]);assert key not in seen;seen.add(key)
        p=float(current[4]);assert math.isfinite(p)
        if current[5] in ['AUROC','AUPRC']:assert 0<=p<=1
        if p!=float(prior[4]):changed[current[0]]+=1
        bytask[current[0]].append((float(prior[4]),p))
    assert dict(changed)==proof['changed_tasks'] and len(bytask)==22
    predictions=json.loads((ROOT/'admet_model_outputs_v24/test_predictions_compact.json').read_text())
    for record in analysis['tasks']:
        task=record['task'];weight=record['weight'];oldp=[a for a,b in bytask[task]];actual=[b for a,b in bytask[task]]
        if not weight:assert oldp==actual;continue
        candidate=predictions[task]
        if record['metric']!='MAE':oldp,candidate=ranks(oldp),ranks(candidate)
        expected=[(1-weight)*a+weight*b for a,b in zip(oldp,candidate)]
        assert len(expected)==len(actual) and all(abs(x-y)<1e-12 for x,y in zip(expected,actual))
    result=dict(status='INDEPENDENT_LOCAL_REPLAY_PASSED',sha256=sha(new),rows=16379,columns=6,tasks=22,
                bytes=len(data),changed_tasks=dict(changed),bom=True,crlf=True,template_identity_exact=True,
                prediction_fusion_replayed=True,local_training=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
