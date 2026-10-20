"""Standard-library-only independent check of actual template rows and predictions."""
import csv,io,json,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def rank(x):
    order=sorted(range(len(x)),key=x.__getitem__);result=[0.]*len(x);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and x[order[i]]==x[order[j]]:j+=1
        for ii in order[i:j]:result[ii]=(i+j)/(2*len(x))
        i=j
    return result

def main():
    dest=ROOT/'admet_submissions/v60';file=next(dest.glob('*.csv'));raw=file.read_bytes();rows=read(file)
    template=read(ROOT/'admet_official/submission_template.csv');basefile=next((ROOT/'admet_submissions/v57').glob('*.csv'));base=read(basefile)
    meta=json.loads((dest/'verification.json').read_text(encoding='utf-8'));replay=json.loads((ROOT/'admet_model_outputs_v60/fusion_replay.json').read_text(encoding='utf-8'))
    assert len(rows)==len(template)==len(base)==16380 and rows[0]==template[0]
    assert hashlib.sha256(raw).hexdigest()==meta['sha256'];assert hashlib.sha256(basefile.read_bytes()).hexdigest()==meta['baseline_sha256']
    assert raw.startswith(b'\xef\xbb\xbf') and b'\n' not in raw.replace(b'\r\n',b'')
    keys=set();changed=Counter()
    for a,t,z in zip(rows[1:],template[1:],base[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:];key=(a[0],a[1]);assert key not in keys;keys.add(key)
        p=float(a[4]);assert math.isfinite(p)
        if a[5] in ['AUROC','AUPRC']:assert 0<=p<=1
        if a[0] not in replay:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    for task,r in replay.items():
        a=[float(x[4]) for x in base[1:] if x[0]==task];actual=[float(x[4]) for x in rows[1:] if x[0]==task];assert a==r['baseline']
        c=r['branch'];w=r['selection']['weight']
        if r['metric']!='MAE':a,c=rank(a),rank(c)
        result=[(1-w)*x+w*y for x,y in zip(a,c)];assert actual==r['result']
        assert max(abs(x-y) for x,y in zip(actual,result))<1e-6
    assert dict(changed)==meta['changed_tasks']
    audit=dict(status='PASS',rows=16379,columns=6,tasks=22,bytes=len(raw),sha256=meta['sha256'],nonprediction_fields_exact=True,
        finite_predictions=True,independent_fusion_replay=True,changed_tasks=dict(changed),changed_rows=sum(changed.values()),
        encoding='UTF-8 BOM',line_endings='CRLF',official_score=None,local_gpu_training=False,automatically_uploaded=False)
    (dest/'local_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
