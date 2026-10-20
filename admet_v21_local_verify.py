"""Local CPU-only verification of remote v21 export and exact fusion replay."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))


def rank(values):
    order=sorted(range(len(values)),key=lambda i:values[i])
    result=[0.]*len(values)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:
            end+=1
        for i in range(start,end):
            result[order[i]]=(start+end)/2/len(values)
        start=end
    return result


def replay(base,candidate,w,mode,metric):
    if mode=='raw':
        return [(1-w)*x+w*y for x,y in zip(base,candidate)]
    q=[(1-w)*x+w*y for x,y in zip(rank(base),rank(candidate))]
    if metric in ['AUROC','AUPRC']:
        return q
    ordered=sorted(base)
    n=len(base)
    result=[]
    for v in q:
        position=max(0,min(n-1,v*n-.5))
        lower=int(math.floor(position)); upper=min(n-1,lower+1)
        result.append(ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower]))
    return result


def main():
    old=next((ROOT/'admet_submissions/v20').glob('*.csv'))
    new=next((ROOT/'admet_submissions/v21').glob('*.csv'))
    proof=read(new.parent/'verification.json')
    assert sha(old)==proof['baseline_sha256']=='7c4001cf13f7cd98f714575a1dce12a827ac7c9800cb65ec163c4f5826587789'
    assert sha(new)==proof['sha256']
    assert proof['status']=='VERIFIED_REMOTE'
    a,b,t=rows(new),rows(old),rows(ROOT/'admet_official/submission_template.csv')
    assert len(a)==len(b)==len(t)==16380
    assert a[0]==b[0]==t[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    blob=new.read_bytes()
    assert blob.startswith(b'\xef\xbb\xbf') and blob.count(b'\r\n')==blob.count(b'\n')==16380
    changes=Counter(); counts=Counter(); actual={}; baseline={}; keys=set()
    for r,prior,official in zip(a[1:],b[1:],t[1:]):
        assert len(r)==6 and r[:4]+r[5:]==prior[:4]+prior[5:]==official[:4]+official[5:]
        key=(r[0],r[1]); assert key not in keys; keys.add(key)
        value=float(r[4]); assert math.isfinite(value)
        if r[5] in ['AUROC','AUPRC']: assert 0<=value<=1
        assert int(r[1])==counts[r[0]]
        counts[r[0]]+=1
        actual.setdefault(r[0],[]).append(value)
        baseline.setdefault(r[0],[]).append(float(prior[4]))
        if r[4]!=prior[4]:changes[r[0]]+=1
    assert len(counts)==22 and set(changes)==set(proof['changed'])
    for audit in proof['audits']:
        assert audit['alignment_passed'] and audit['all_stages_verified'] and audit['fold_scaffold_overlap']==0
        s=audit['selection']; task=audit['task']
        if s['weight']==0:
            assert actual[task]==baseline[task]
            continue
        assert audit['model_replay_passed']
        path=ROOT/'admet_model_outputs_v21'/task/'finetune_test.csv'
        candidate=[float(r[0]) for r in rows(path)[1:]]
        expected=replay(baseline[task],candidate,s['weight'],s['mode'],audit['metric'])
        assert len(expected)==len(actual[task])
        assert all(abs(x-y)<=1e-10*max(1,abs(y)) for x,y in zip(actual[task],expected))
    result=dict(status='VERIFIED_LOCAL_AND_REMOTE',submission=str(new.resolve()),sha256=sha(new),
                bytes=len(blob),rows=16379,columns=6,tasks=22,encoding='UTF-8 BOM',line_endings='CRLF',
                changed_tasks=dict(changes),changed_rows=sum(changes.values()),
                baseline='v20',baseline_score=50.019727,platform_score=None,
                template_metadata_exact=True,fusion_independently_replayed=True,
                local_gpu_training=False,automatic_submission=False)
    (new.parent/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    current=read(ROOT/'admet_current_candidate.json')
    current.update(recommended_release='v21_candidate',submission=str(new.resolve()),sha256=sha(new),
                   experimental_release='v21',experimental_submission=str(new.resolve()),experimental_sha256=sha(new),
                   candidate_platform_score=None,platform_anchor_release='v20',platform_anchor_score=50.019727,
                   awaiting='User platform evaluation of v21',report='admet_diagnostics/v21_delivery_report.md',
                   verification='admet_submissions/v21/local_verification.json',target_achieved=False)
    (ROOT/'admet_current_candidate.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
