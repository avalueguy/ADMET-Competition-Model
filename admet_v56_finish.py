"""Validate frozen new-model choices, refit, calibrate supported tasks, export one candidate."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(_key,'2')
import csv
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
import admet_v24_models as b
import admet_v27_models as graph
import admet_v44_run as pair
import admet_v51_run as trees
import admet_v52_neural as neural
import admet_v54_graph as adapt
from admet_v52_calibration import fit_lad

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v56'


def signature():
    return hashlib.sha256((b.sha(ROOT/'admet_v56_finish.py')+trees.signature()+neural.signature()+adapt.signature()).encode()).hexdigest()


def wait_inputs():
    sources=[neural.OUT/'analysis.json',adapt.OUT/'analysis.json',ROOT/'admet_model_outputs_v55_nested_calibration/analysis.json']
    start=time.time()
    while True:
        statuses=[]
        for p in sources:
            try:statuses.append(json.loads(p.read_text()).get('status'))
            except (FileNotFoundError,json.JSONDecodeError):statuses.append(None)
        if statuses==['FROZEN','FROZEN','COMPLETE']:return
        if time.time()-start>10800:raise TimeoutError(dict(inputs=statuses))
        time.sleep(30)


def fresh_models(d,c,g):
    result=[]
    for family,source in [('neural',neural.OUT),('graph',adapt.OUT)]:
        report=json.loads((source/'analysis.json').read_text());assert report['status']=='FROZEN'
        for record in report['tasks']:
            s=record['selection']
            if not s:continue
            task=record['task'];k=d['tasks'].index(task);m=d['metrics'][k];parts=[]
            for seed in [561016,561017]:
                if family=='neural':
                    allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows])
                    held=np.random.default_rng(seed+k).permutation(groups)[:max(1,int(.25*len(groups)))]
                    active=np.isin(d['rg'][allrows],held);rr,q=allrows[~active],allrows[active]
                    pp=neural.variants(neural.fit(d,c,k,rr,q,source/task/f'v56_fresh_{seed}'))[s['variant']]
                    controls=pair.variants(pair.fit(d,k,rr,q,pair.OUT/task/f'v56_control_{seed}'),m)
                    old=json.loads((pair.OUT/'analysis.json').read_text())
                    sel=next(x for x in old['tasks'] if x['task']==task)['selection']
                    ctrl=controls[sel['variant']] if sel else controls[0]
                    provenance='New scaffold repartition; freshly refitted v44 selected source, not full ensemble'
                else:
                    # A fresh source checkpoint would otherwise have seen the new holdout.
                    # Draw groups only from an existing excluded source fold; add allowed
                    # train/valid target labels to adaptation while retaining source exclusion.
                    fold=0 if seed==561016 else 1;f=np.load(b.OUT/'folds.npy')
                    pool=np.flatnonzero((d['tid']==k)&(f==fold));groups=np.unique(d['rg'][pool])
                    held=np.random.default_rng(seed+k).permutation(groups)[:max(1,int(.7*len(groups)))]
                    q=pool[np.isin(d['rg'][pool],held)]
                    rr=np.flatnonzero((d['tid']==k)&(d['split']<2)&~np.isin(d['rg'],held))
                    partition='cv/fold'+str(fold)
                    pp=adapt.variants(adapt.fit(d,g,k,rr,q,partition,source/task/f'v56_fresh_{seed}'),m)[s['variant']]
                    original=[]
                    for modelseed in adapt.P['seeds']:
                        z=np.load(graph.OUT/partition/str(modelseed)/'predictions.npz')
                        order=np.argsort(z['query']);queries=z['query'][order];loc=np.searchsorted(queries,q)
                        np.testing.assert_array_equal(queries[loc],q)
                        original.append(z['p'][:,order[loc]])
                    original=np.mean(original,0)
                    ctrl=adapt.variants(original,m)[s['variant']]
                    provenance='Resampled groups from checkpoint-excluded fold; extra allowed target fit labels; original graph source comparator, not full ensemble'
                merged=b.blend(m,ctrl,pp,s['weight']);gain=b.delta(m,d['y'][q],ctrl,merged)
                parts.append(dict(seed=seed,fit_rows=len(rr),query_rows=len(q),metric=m,
                    control=b.metric(m,d['y'][q],ctrl),candidate=b.metric(m,d['y'][q],merged),gain=gain,
                    query_label_seen_by_model=False,scaffold_overlap=0,provenance=provenance))
                b.emit('v56_fresh_source',task=task,family=family,**parts[-1])
            gains=[x['gain'] for x in parts]
            result.append(dict(family=family,task=task,selection=s,parts=parts,mean_gain=float(np.mean(gains)),
                pass_gate=bool(np.mean(gains)>0 and min(gains)>-.025)))
            b.save(OUT/'fresh_models.json',dict(status='IN_PROGRESS',signature=signature(),tasks=result))
    b.save(OUT/'fresh_models.json',dict(status='COMPLETE',signature=signature(),tasks=result))
    return result


def final_export(d,c,g,fresh):
    source=next((ROOT/'admet_submissions/v46').glob('*.csv'));assert b.sha(source)==trees.P['baseline_sha256']
    def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
    rows=read(source);original=[r.copy() for r in rows];template=read(ROOT/'admet_official/submission_template.csv')
    test={};cv={};valid={};replay={};selected={}
    basev,basec=trees.baseline(d);folds=np.load(b.OUT/'folds.npy')
    for k,task in enumerate(d['tasks']):
        pos=[i for i,row in enumerate(rows) if i and row[0]==task]
        test[task]=np.array([float(rows[i][4]) for i in pos])
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        cv[task]=basec[tr].copy();valid[task]=basev[vi].copy()
    # One new model source per endpoint; compare CV utility if two families pass.
    candidates={}
    for record in fresh:
        if record['pass_gate']:candidates.setdefault(record['task'],[]).append(record)
    for task,choices in candidates.items():
        winner=max(choices,key=lambda x:x['selection']['cv_mean'])
        k=d['tasks'].index(task);m=d['metrics'][k];s=winner['selection'];family=winner['family']
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        rr=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        if family=='neural':
            out=neural.OUT;raw=neural.fit(d,c,k,rr,q,out/task/'final');p=neural.variants(raw)[s['variant']]
            vp=neural.variants(neural.fit(d,c,k,tr,vi,out/task/'valid'))[s['variant']]
        else:
            out=adapt.OUT;raw=adapt.fit(d,g,k,rr,q,'final',out/task/'final');p=adapt.variants(raw,m)[s['variant']]
            vp=adapt.variants(np.load(out/task/'valid/predictions.npz')['p'],m)[s['variant']]
        cp=np.load(out/task/'cv_predictions.npy')[s['variant'],tr]
        old=test[task].copy();test[task]=b.blend(m,old,p,s['weight'])
        valid[task]=b.blend(m,valid[task],vp,s['weight'])
        for f in range(3):
            active=folds[tr]==f;cv[task][active]=b.blend(m,cv[task][active],cp[active],s['weight'])
        selected[task]=dict(family=family,selection=s,fresh=winner)
        replay[task]=dict(metric=m,baseline=old.tolist(),branch=p.tolist(),weight=s['weight'],raw=raw.tolist(),family=family,
            model_result=test[task].tolist(),calibration=None)
        b.emit('v56_final_model',task=task,family=family)
    calibration=json.loads((ROOT/'admet_model_outputs_v55_nested_calibration/analysis.json').read_text())
    for record in calibration['tasks']:
        if not record['pass_gate']:continue
        task=record['task'];k=d['tasks'].index(task);tr=np.flatnonzero((d['tid']==k)&(d['split']==0))
        # Re-evaluate calibrated errors if this endpoint also has a newly selected model.
        y=d['y'][tr];p=cv[task];ff=folds[tr];corrected=np.empty_like(p)
        for f in range(3):
            active=ff!=f;query=~active
            beta=np.array([1.,np.median(y[active]-p[active])]) if record['mode']=='median_bias' else fit_lad(p[active],y[active])
            beta=np.array([1.,0.])+record['shrink']*(beta-np.array([1.,0.]))
            corrected[query]=beta[0]*p[query]+beta[1]
        if task=='ppbr_az':corrected=corrected.clip(0,100)
        gains=[b.delta('MAE',y[ff==f],p[ff==f],corrected[ff==f]) for f in range(3)]
        if np.mean(gains)<=0 or sum(g>0 for g in gains)<2:continue
        beta=np.array([1.,np.median(y-p)]) if record['mode']=='median_bias' else fit_lad(p,y)
        beta=np.array([1.,0.])+record['shrink']*(beta-np.array([1.,0.]))
        old=test[task].copy();test[task]=beta[0]*old+beta[1]
        if task=='ppbr_az':test[task]=test[task].clip(0,100)
        if task not in replay:replay[task]=dict(metric='MAE',baseline=old.tolist(),branch=None,weight=0.,family=None,model_result=old.tolist())
        replay[task]['calibration']=dict(coefficients=beta.tolist(),bounds=[0,100] if task=='ppbr_az' else None,
            cv_gains=gains,nested_evaluation=record)
        selected.setdefault(task,{})['calibration']=dict(coefficients=beta.tolist(),cv_gains=gains)
    changed={}
    for task in replay:
        pos=[i for i,row in enumerate(rows) if i and row[0]==task]
        for i,value in zip(pos,test[task]):rows[i][4]=repr(float(value))
        replay[task]['result']=test[task].tolist()
        changed[task]=sum(rows[i][4]!=original[i][4] for i in pos)
    assert changed and len(rows)==len(template)==16380 and rows[0]==template[0]
    for a,t in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==t[:4]+t[5:] and np.isfinite(float(a[4]))
        if a[5] in ['AUROC','AUPRC']:assert 0<=float(a[4])<=1
    dest=ROOT/'admet_submissions/v56';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261016.csv'
    buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\r\n').writerows(rows);raw=b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8')
    if path.exists():assert path.read_bytes()==raw,'Never overwrite an existing different submission'
    else:path.write_bytes(raw)
    b.save(OUT/'fusion_replay.json',replay)
    b.save(OUT/'release.json',dict(signature=signature(),selected=selected,baseline_sha256=b.sha(source),
        limitations='Known development labels reused; old ensemble CV proxy, fresh source comparisons and nested calibration are not independent platform evaluation. Unknown official formula and score.'))
    b.save(dest/'verification.json',dict(status='EXPERIMENTAL_AWAITING_PLATFORM',sha256=b.sha(path),signature=signature(),
        rows=16379,columns=6,tasks=22,bytes=len(raw),baseline='v46',baseline_sha256=b.sha(source),current_official_score=None,
        changed_tasks=changed,local_gpu_training=False,automatically_uploaded=False,encoding='UTF-8 BOM',line_endings='CRLF'))
    b.emit('v56_export',path=str(path),sha256=b.sha(path),changed_tasks=changed)


def main():
    assert os.name!='nt';OUT.mkdir(exist_ok=True);wait_inputs();b.remote_only();torch.backends.cuda.matmul.allow_tf32=False
    d=b.build_data();c=trees.chemistry(d);g=graph.graphs(d)
    fresh=fresh_models(d,c,g);final_export(d,c,g,fresh)
    b.save(OUT/'completed.json',dict(status='COMPLETE',signature=signature()))


if __name__=='__main__':main()
