"""Combine confirmed new endpoint models and measured historical recovery."""
import csv,io,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v119_matched_half_life as matched
import v121_conditional_ranking as ranking
import v123_normalized_rankmap as normalized
import v116_shift_weighting as shifted
from repack_upload_template import field_spans
from verify_submission import verify
b,r83=matched.b,matched.r83
OUT=ROOT/'admet_model_outputs_v124'
BASE=next((ROOT/'admet_submissions/v114_herg_history52').glob('*.csv'))
BASE_SHA='4b23b55764d6620d7f31171f221412d682d5cd53bd7352364dbed4359dfc167e'
HISTORY=next((ROOT/'admet_submissions/v112_history_expansion').glob('*.csv'))
HISTORY_SHA='0ca9e57d41c4fa3855a97560ee42d6419654530fa5013604f7ca2e39f2ca3af5'

def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def choose():
    choices={};experiments=[]
    for name,v in [('v116',shifted),('v121',ranking),('v123',normalized)]:
        f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
        assert c['status']=='COMPLETE' and f['signature']==c['signature']==v.sig()
        accepted=[r for r in c['records'] if r['accepted']];assert c['accepted']==[r['task'] for r in accepted]
        experiments.append(dict(version=name,accepted=c['accepted'],confirmation_sha256=b.sha(v.OUT/'confirmation.json')))
        for r in accepted:
            assert any(x['task']==r['task'] and x['parts']==r['parts'] for x in f['selected'])
            objective=min(p['gain'] for p in r['parts'])
            if r['task'] not in choices or objective>choices[r['task']]['objective']:
                choices[r['task']]=dict(version=name,record=r,objective=objective,weight=v.WEIGHT)
    f=json.loads((matched.OUT/'frozen_selection.json').read_text());c=json.loads((matched.OUT/'confirmation.json').read_text())
    audit=json.loads((matched.OUT/'independent_verification.json').read_text())
    assert f['signature']==c['signature']==audit['signature']==matched.sig() and audit['status']=='PASS'
    assert c['accepted'] and f['selected'] and c['record']==f['record']
    choices['half_life_obach']=dict(version='v119',record=c,objective=f['record']['objective'],weight=matched.WEIGHT)
    experiments.append(dict(version='v119',accepted=['half_life_obach'],confirmation_sha256=b.sha(matched.OUT/'confirmation.json'),
                            independent_verification_sha256=b.sha(matched.OUT/'independent_verification.json')))
    return choices,experiments

def assemble():
    assert b.sha(BASE)==BASE_SHA and b.sha(HISTORY)==HISTORY_SHA
    choices,experiments=choose();d=matched.v.load();out=rows(BASE);original=[r.copy() for r in out];history=rows(HISTORY)
    assert len(out)==len(history)==16380
    sources={};checks=[]
    if 'ames' not in choices:
        for a,z in zip(out[1:],history[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[0]=='ames':a[4]=z[4]
        sources['ames']=dict(kind='measured_historical_task',source='v112',source_sha256=HISTORY_SHA,
            previous_official_task_score=49.980787,source_official_task_score=50.059239,
            retrospective_selection=True,expected_total_delta_if_repeated=.078452/22)
    for task,ch in choices.items():
        k=d['tasks'].index(task);m=d['metrics'][k];w=ch['weight'];name=ch['version'];r=ch['record']
        for part in ('fold0','fold1','fold2','valid','final'):
            if name=='v119':
                z=matched.predict(d,k,part,r['record']['view']);fit,q=matched.split(d,k,part)
                a=matched.base(d,k,part);records=r['parts']
            else:
                v={'v116':shifted,'v121':ranking,'v123':normalized}[name]
                z=v.predict(d,k,part)[0] if name=='v116' else v.predict(d,k,part)
                fit,q=r83.split(d,k,part);a=matched.v.base(d,k,part) if part!='final' else None;records=r['confirmation']
            if part=='final':
                pos=[i for i,a in enumerate(out) if i and a[0]==task];a=np.array([float(out[i][4]) for i in pos])
            p=(1-w)*a+w*z if m=='MAE' else (1-w)*r83.rankdata(a)/len(a)+w*r83.rankdata(z)/len(z)
            assert np.isfinite(p).all() and len(p)==len(q)
            if part=='final':
                for i,x in zip(pos,p):out[i][4]=repr(float(x))
            else:
                expected=next(x for x in records if x['part']==part);actual=r83.score(m,d['y'][q],p)
                np.testing.assert_allclose(actual,expected['candidate'],atol=1e-12,rtol=0)
                checks.append(dict(task=task,version=name,part=part,rows=len(q),gain=expected['gain'],candidate=actual))
        sources[task]=dict(kind='new_trained_model',**ch)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();sp=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(out[1:],1):
        lo,hi=sp[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);payload=b''.join(chunks)
    folder=ROOT/'admet_submissions/v124_matched_models_history';folder.mkdir(exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv'
    if file.exists():assert file.read_bytes()==payload,'Refuse overwriting a different candidate.'
    else:file.write_bytes(payload)
    fmt=verify(file,template);ss=field_spans(payload);back=[];cursor=0
    for i in range(1,len(out)):
        lo,hi=ss[i][4];a,z=sp[i][4];back.extend((payload[cursor:lo],raw[a:z]));cursor=hi
    back.append(payload[cursor:]);assert b''.join(back)==raw
    changed=Counter(a[0] for a,z in zip(out[1:],original[1:]) if a[4]!=z[4]);assert set(changed)==set(sources)
    for a,z in zip(out[1:],original[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in sources:assert a==z
    fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),model_validation_replay='PASS')
    report=dict(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',version='v124',baseline='v114',baseline_official_score=51.091,
        baseline_sha256=BASE_SHA,file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),sources=sources,experiments=experiments,
        validation_replay=checks,verification=fmt,official_score=None,target_52_achieved=False,
        new_models_official_data_only=True,test_labels_used=False,automatically_uploaded=False,
        limitations='Public development labels reused; historical recovery retrospective. Half-life comparison replays exact v114 branch on matched fit rows; other tasks use approximate historical proxies. Official improvement unconfirmed.')
    b.save(OUT/'evidence.json',report);b.save(OUT/'verification.json',fmt);b.save(folder/'model_evidence.json',report)
    b.save(file.with_suffix('.verification.json'),fmt)
    print(json.dumps(dict(status=report['status'],file=report['file'],sha256=report['sha256'],changed_tasks=dict(changed))),flush=True)

if __name__=='__main__':assemble()
