"""Independent, standard-library-only verification of the delivered submission."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(p):
    return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))


def main():
    dest=ROOT/'admet_submissions/v51'
    file=next(dest.glob('*.csv'));raw=file.read_bytes()
    manifest=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
    source=next((ROOT/'admet_submissions/v46').glob('*.csv'))
    template=read(ROOT/'admet_official/submission_template.csv');rows=read(file);old=read(source)
    replay=json.loads((ROOT/'admet_model_outputs_v51/fusion_replay.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['baseline_sha256']
    assert raw.startswith(b'\xef\xbb\xbf') and b'\n' not in raw.replace(b'\r\n',b'')
    assert len(rows)==len(template)==len(old)==16380 and rows[0]==template[0]
    keys=set();changed=Counter();values={}
    for a,t,z in zip(rows[1:],template[1:],old[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:]
        key=(a[0],a[1]);assert key not in keys;keys.add(key)
        v=float(a[4]);assert math.isfinite(v)
        if a[5] in ['AUROC','AUPRC']:assert 0<=v<=1
        if a[0] not in replay:assert a==z
        if a[4]!=z[4]:changed[a[0]]+=1
        values.setdefault(a[0],[]).append(v)
    for task,record in replay.items():
        spec=record['selection'];v=spec['variant'];pp=record['source'];weight=spec['weight']
        if v<3:branch=pp[v]
        elif v==3:branch=[(a+b)/2 for a,b in zip(pp[0],pp[1])]
        elif v==4:branch=[(a+b)/2 for a,b in zip(pp[1],pp[2])]
        else:raise AssertionError(v)
        result=[(1-weight)*a+weight*b for a,b in zip(record['old'],branch)]
        assert values[task]==record['result']
        assert max(abs(a-b) for a,b in zip(result,values[task]))<1e-12
    assert dict(changed)==manifest['changed_tasks']
    result=dict(status='PASS',sha256=manifest['sha256'],rows=16379,columns=6,
        tasks=len(Counter(a[0] for a in rows[1:])),bytes=len(raw),
        changed_tasks=dict(changed),changed_rows=sum(changed.values()),
        nonprediction_fields_exact=True,predictions_independently_replayed=True,
        finite_predictions=True,encoding='UTF-8 BOM',line_endings='CRLF',
        local_gpu_training=False,automatic_upload=False,current_official_score=None)
    (dest/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
