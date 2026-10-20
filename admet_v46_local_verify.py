"""Standard-library-only submission and independent rank-blending verification."""
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v46'

def read(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))

def rank(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    result=[0.]*len(values)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        percentile=(i+j)/(2*len(values))
        for a in order[i:j]:result[a]=percentile
        i=j
    return result

def main():
    file=next((ROOT/'admet_submissions/v46').glob('*.csv'))
    raw=file.read_bytes()
    rows=read(file)
    template=read(ROOT/'admet_official/submission_template.csv')
    source=read(OUT/'baseline_v45.csv')
    report=json.loads((file.parent/'verification.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==report['sha256']
    assert len(rows)==len(template)==len(source)==16380
    assert rows[0]==template[0]==source[0]
    assert raw.startswith(b'\xef\xbb\xbf')
    assert b'\n' not in raw.replace(b'\r\n',b'')
    keys=set();changed=Counter()
    replay=json.loads((OUT/'fusion_replay.json').read_text(encoding='utf-8'))
    for row,trow,old in zip(rows[1:],template[1:],source[1:]):
        assert len(row)==6 and row[:4]+row[5:]==trow[:4]+trow[5:]
        assert (row[0],row[1]) not in keys
        keys.add((row[0],row[1]))
        value=float(row[4]);assert math.isfinite(value)
        if row[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if row[0] not in replay:assert row==old
        if row[4]!=old[4]:changed[row[0]]+=1
    for task,spec in replay.items():
        p=[rank(a) for a in spec['source']]
        if spec['variant']<4:branch=p[spec['variant']]
        elif spec['variant']==4:branch=[sum(v)/2 for v in zip(*p[2:])]
        else:branch=[sum(v)/4 for v in zip(*p)]
        a,c=rank(spec['baseline']),rank(branch)
        expected=[(1-spec['weight'])*x+spec['weight']*y for x,y in zip(a,c)]
        actual=[float(row[4]) for row in rows[1:] if row[0]==task]
        assert actual==spec['result']
        assert max(abs(x-y) for x,y in zip(expected,actual))<1e-14,task
    result=dict(status='PASS',sha256=report['sha256'],baseline_sha256=report['baseline_sha256'],
        rows=len(rows)-1,columns=6,tasks=len(Counter(r[0] for r in rows[1:])),
        bytes=len(raw),changed_tasks=dict(changed),changed_rows=sum(changed.values()),
        template_nonprediction_fields_exact=True,rank_blending_independently_replayed=True,
        encoding='UTF-8 BOM',line_endings='CRLF',finite_predictions=True,
        classification_range_valid=True,local_gpu_training=False,automatically_uploaded=False)
    (file.parent/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
