"""Verify frozen v168 models independently and export only confirmed task changes."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import json,sys,time
from collections import Counter
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v168_sparse_linear as model
import v148_experiment_runner as runner
from v161_verify_and_assemble import same,table
from repack_upload_template import field_spans
from verify_submission import verify
v,b,r83=model.v,model.b,model.r83
OUT=ROOT/'admet_model_outputs_v169'

def main():
    OUT.mkdir(exist_ok=True)
    for _ in range(720):
        cp=model.OUT/'confirmation.json'
        if cp.exists() and json.loads(cp.read_text())['status']=='COMPLETE':break
        time.sleep(15)
    else:raise TimeoutError('v168 training did not complete; preserve partial results')
    confirmed=json.loads(cp.read_text());frozen=json.loads((model.OUT/'frozen_selection.json').read_text());assert confirmed['signature']==frozen['signature']==model.sig()
    selected={r['task']:r for r in confirmed['records'] if r['accepted']}
    if not selected:
        b.save(OUT/'no_candidate.json',dict(status='NO_CONFIRMED_NEW_MODEL',training_version='v168',baseline='v145',submission_created=False,official_baseline_score=51.166,confirmation_sha256=b.sha(cp)))
        print('NO_CONFIRMED_NEW_MODEL',flush=True);return
    d=model.load();original=table(v.ANCHOR);rows=[r.copy() for r in original];checks=[]
    for task,r in selected.items():
        k=d['tasks'].index(task);chosen=next(ch for ch in frozen['selected'] if ch['task']==task)
        assert runner.confirm(d,k,model.predict,chosen)==r
        for part in v.PARTS:
            model.predict(d,k,part,r['config']);a=model.design(d,k,part);folder=model.OUT/'models'/task/part/model.key(r['config']);cached=np.load(folder/'predictions.npz');assert str(cached['signature'])==model.sig()
            transform=joblib.load(folder/'transform.joblib');same(transform,{key:value for key,value in a.items() if key not in ('x','qx')});same(cached['fit'],a['fit']);same(cached['query'],a['query'])
            replay=joblib.load(folder/'model.joblib');cls=d['metrics'][k] in ('AUROC','AUPRC');raw=replay.decision_function(a['qx']) if cls else replay.predict(a['qx']);p=expit(raw) if cls else raw*float(cached['scale'])+float(cached['center'])
            if task=='ppbr_az':p=p.clip(0,100)
            np.testing.assert_array_equal(p,cached['p']);saved=d['y'][a['query']].copy()
            try:
                d['y'][a['query']]=987654.;d.pop('_v96_assays',None);same(a,model.design(d,k,part))
            finally:d['y'][a['query']]=saved;d.pop('_v96_assays',None)
            candidate=v.mix(d['metrics'][k],v.baseline(d,k,part),p,r['weight']);check=dict(task=task,part=part,model_reload=True,query_label_perturbation_invariant=True,model_sha256=b.sha(folder/'model.joblib'),transform_sha256=b.sha(folder/'transform.joblib'))
            if part=='final':
                positions=[i for i,z in enumerate(original) if i and z[0]==task];assert len(positions)==len(candidate)
                for i,z in zip(positions,candidate):rows[i][4]=repr(float(z))
            else:
                expected=next(x for x in r['confirmation'] if x['part']==part);np.testing.assert_allclose(r83.score(d['metrics'][k],d['y'][a['query']],candidate),expected['candidate'],atol=1e-12,rtol=0)
            checks.append(check)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();spans=field_spans(raw);pieces=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=spans[i][4];pieces.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    pieces.append(raw[cursor:]);payload=b''.join(pieces);folder=ROOT/'admet_submissions/v169_sparse_linear';folder.mkdir(exist_ok=True);file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():assert file.read_bytes()==payload
    else:file.write_bytes(payload)
    fmt=verify(file,template);changed=Counter()
    for old,new in zip(original[1:],rows[1:]):
        assert old[:4]+old[5:]==new[:4]+new[5:]
        if old[0] not in selected:assert old==new
        if old[4]!=new[4]:changed[old[0]]+=1
    assert set(changed)==set(selected) and 'hia_hou' not in changed and fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only']
    outspans=field_spans(payload);pieces=[];cursor=0
    for i in range(1,len(rows)):
        lo,hi=outspans[i][4];left,right=spans[i][4];pieces.extend((payload[cursor:lo],raw[left:right]));cursor=hi
    pieces.append(payload[cursor:]);assert b''.join(pieces)==raw
    fmt.update(nonprediction_bytes_exact=True,protected_task_predictions_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed));b.save(file.with_suffix('.verification.json'),fmt)
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v169',training_version='v168',baseline='v145',baseline_official_score=51.166,baseline_sha256=v.ANCHOR_SHA,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),selected=selected,checks=checks,verification=fmt,official_score=None,automatically_uploaded=False,limitations='Public labels reused; historical controls partly approximate. No official gain guaranteed.')
    b.save(OUT/'evidence.json',report);b.save(folder/'model_evidence.json',report);b.save(OUT/'independent_verification.json',dict(status='PASS',checks=checks,template_nonprediction_bytes_exact=True));print(json.dumps(dict(status=report['status'],changed_tasks=dict(changed),sha256=report['sha256'])),flush=True)
if __name__=='__main__':main()
