"""Independent no-GPU submission verification, including rank fusion and fitted calibration."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))


def rank(values):
    order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for ix in order[i:j]:out[ix]=(i+j)/(2*len(values))
        i=j
    return out


def main():
    dest=ROOT/'admet_submissions/v56';file=next(dest.glob('*.csv'));raw=file.read_bytes()
    meta=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    replay=json.loads((ROOT/'admet_model_outputs_v56/fusion_replay.json').read_text(encoding='utf-8'))
    source=next((ROOT/'admet_submissions/v46').glob('*.csv'))
    assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['baseline_sha256']
    rows,old,template=read(file),read(source),read(ROOT/'admet_official/submission_template.csv')
    assert len(rows)==len(old)==len(template)==16380 and rows[0]==template[0]
    assert raw.startswith(b'\xef\xbb\xbf') and b'\n' not in raw.replace(b'\r\n',b'')
    keys=set();changed=Counter()
    for a,z,t in zip(rows[1:],old[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:]
        assert (a[0],a[1]) not in keys;keys.add((a[0],a[1]))
        p=float(a[4]);assert math.isfinite(p)
        if a[5] in ['AUROC','AUPRC']:assert 0<=p<=1
        if a[0] not in replay:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
    for task,s in replay.items():
        base=s['baseline'];branch=s['branch'];weight=s['weight']
        actual_base=[float(a[4]) for a in old[1:] if a[0]==task]
        assert base==actual_base
        if branch is None:expected=base[:]
        else:
            a,c=(base,branch) if s['metric']=='MAE' else (rank(base),rank(branch))
            expected=[(1-weight)*x+weight*y for x,y in zip(a,c)]
        assert max(abs(a-b) for a,b in zip(expected,s['model_result']))<1e-6
        if s.get('calibration'):
            a,b=s['calibration']['coefficients'];bounds=s['calibration']['bounds']
            expected=[a*x+b for x in expected]
            if bounds:expected=[min(bounds[1],max(bounds[0],x)) for x in expected]
        actual=[float(a[4]) for a in rows[1:] if a[0]==task]
        assert actual==s['result']
        assert max(abs(a-b) for a,b in zip(actual,expected))<1e-6
    assert dict(changed)==meta['changed_tasks']
    result=dict(status='PASS',sha256=meta['sha256'],rows=16379,columns=6,tasks=22,bytes=len(raw),
        changed_tasks=dict(changed),changed_rows=sum(changed.values()),nonprediction_fields_exact=True,
        independent_fusion_and_calibration_replay=True,finite_predictions=True,
        encoding='UTF-8 BOM',line_endings='CRLF',current_official_score=None,local_gpu_training=False,automatically_uploaded=False)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
