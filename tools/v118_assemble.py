"""Assemble frozen, confirmed task models with verified history over v114."""
from __future__ import annotations
import csv,io,json,sys,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v115_diverse_specialists as sequence
import v116_shift_weighting as shifted
import v117_conditional_structure as conditional
from repack_upload_template import field_spans
b,r83=sequence.b,sequence.r83
OUT=ROOT/'admet_model_outputs_v118'
BASE=next((ROOT/'admet_submissions/v114_herg_history52').glob('*.csv'))
BASE_SHA='4b23b55764d6620d7f31171f221412d682d5cd53bd7352364dbed4359dfc167e'
HISTORY=next((ROOT/'admet_submissions/v112_history_expansion').glob('*.csv'))
HISTORY_SHA='0ca9e57d41c4fa3855a97560ee42d6419654530fa5013604f7ca2e39f2ca3af5'

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def save(name,x):b.save(OUT/name,x)

def selected():
    choices={};experiments=[]
    for name,v in [('v115',sequence),('v116',shifted),('v117',conditional)]:
        f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
        assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
        accepted=[r for r in c['records'] if r['accepted']]
        assert c['accepted']==[r['task'] for r in accepted]
        experiments.append(dict(version=name,confirmation_sha256=b.sha(v.OUT/'confirmation.json'),accepted=c['accepted']))
        for r in accepted:
            assert any(x['task']==r['task'] for x in f['selected'])
            # Resolve competing methods using development evidence only. The
            # confirmation data decide acceptance, never which winner to pick.
            objective=min(p['gain'] for p in r['parts'])
            if r['task'] not in choices or objective>choices[r['task']]['objective']:
                choices[r['task']]=dict(version=name,record=r,objective=float(objective))
    return choices,experiments

def assemble():
    assert b.sha(BASE)==BASE_SHA and b.sha(HISTORY)==HISTORY_SHA
    choices,experiments=selected();assert choices,'No newly confirmed models. Do not rename a history-only file.'
    d=sequence.load();rows=read(BASE);original=[r.copy() for r in rows];history=read(HISTORY)
    assert len(rows)==16380 and len(history)==len(rows)
    sources={};checks=[]
    # AMES is an actually observed official improvement in v112; v113 restored
    # it together with the failing hERG. Restore only that positive endpoint.
    if 'ames' not in choices:
        for a,z in zip(rows[1:],history[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[0]=='ames':a[4]=z[4]
        sources['ames']=dict(kind='officially_measured_history',version='v112',source_sha256=HISTORY_SHA,
            previous_official_task_score=49.980787,source_official_task_score=50.059239,
            expected_total_delta_if_reproduced=(50.059239-49.980787)/22)
    for task,ch in choices.items():
        v={'v115':sequence,'v116':shifted,'v117':conditional}[ch['version']]
        k=d['tasks'].index(task);m=d['metrics'][k];r=ch['record'];w=v.WEIGHT
        for part in ('fold0','fold1','fold2','valid','final'):
            if v is sequence:z=v.predict(d,k,part,r['config'])
            elif v is shifted:z=v.predict(d,k,part)[0]
            else:z=v.fit(d,k,part,r['depth'],'full')
            if part=='final':
                positions=[i for i,a in enumerate(rows) if i and a[0]==task]
                a=np.array([float(rows[i][4]) for i in positions])
            else:
                a=sequence.base(d,k,part)
            p=(1-w)*a+w*z if m=='MAE' else (1-w)*r83.rankdata(a)/len(a)+w*r83.rankdata(z)/len(z)
            assert np.isfinite(p).all()
            if m in ('AUROC','AUPRC'):assert np.min(p)>=0 and np.max(p)<=1
            if part=='final':
                assert len(positions)==len(p)
                for i,x in zip(positions,p):rows[i][4]=repr(float(x))
            else:
                q=r83.split(d,k,part)[1];actual=r83.score(m,d['y'][q],p)
                expected=next(z['candidate'] for z in r['confirmation'] if z['part']==part)
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                checks.append(dict(task=task,version=ch['version'],part=part,score=actual,
                    gain=next(z['gain'] for z in r['confirmation'] if z['part']==part)))
        sources[task]=dict(kind='new_trained_model',version=ch['version'],weight=w,record=r)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();sp=field_spans(raw)
    chunks=[];cursor=0
    for i,r in enumerate(rows[1:],1):
        lo,hi=sp[i][4];chunks.extend((raw[cursor:lo],r[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    dest=ROOT/'admet_submissions/v118_diverse_task_models';dest.mkdir(exist_ok=True)
    file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    file.write_bytes(output)
    # verify_submission's legacy optional evidence hook only applies to v66.
    from verify_submission import verify
    verification=verify(file,template)
    ss=field_spans(output);back=[];cursor=0
    for i in range(1,len(rows)):
        lo,hi=ss[i][4];a,z=sp[i][4];back.extend((output[cursor:lo],raw[a:z]));cursor=hi
    back.append(output[cursor:]);assert b''.join(back)==raw
    changed=Counter(a[0] for a,z in zip(rows[1:],original[1:]) if a[4]!=z[4])
    assert set(changed)==set(sources)
    for a,z in zip(rows[1:],original[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in sources:assert a==z
    verification.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),model_validation_replay='PASS')
    save('verification.json',verification)
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v118',baseline='v114',baseline_sha256=BASE_SHA,
        baseline_official_score=51.091,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),
        sources=sources,changed_tasks=dict(changed),experiments=experiments,validation_replay=checks,
        verification=verification,official_score=None,target_52_achieved=False,test_labels_used=False,
        external_data=False,automatically_uploaded=False,
        limitations='Model comparisons use historically reused labels and approximate v101/v85 controls, not full v114 OOF. Official gain is unconfirmed.')
    save('evidence.json',report);b.save(dest/'model_evidence.json',report);b.save(file.with_suffix('.verification.json'),verification)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    sequence.torch.set_num_threads(4)
    sequence.torch.backends.cuda.matmul.allow_tf32=False
    assemble()
