"""Matched residual screening with frozen task-level confirmation.

Reads only official train/valid labels through the audited loader. Test
structures come from the upload template. Every classification ensemble
averages raw logits, and regression ensembles average predictions, at all
stages. Known baseline residual endpoints are replayed on the same folds.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse, csv, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtri
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
from repack_upload_template import field_spans
from verify_submission import verify

OUT=ROOT/'admet_model_outputs_v83'
SEEDS=(834026,834027,834028)
ANCHOR_SHA='6941a4aeb9b817445d8a9df24b93f3a57671883ad814aec6095ef2853e4f3fa0'
ANCHOR=ROOT/'admet_submissions/v82_targeted/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
BASE_SPECS={
    'cyp2c9_veith':(4,300,(802026,802027,802028)),
    'bioavailability_ma':(3,100,(802026,802027,802028)),
    'cyp2d6_veith':(4,300,(802026,802027,802028)),
    'pgp_broccatelli':(3,100,(802026,802027,802028)),
    'cyp2c9_substrate_carbonmangels':(2,100,(824026,824027,824028)),
    'cyp2d6_substrate_carbonmangels':(2,50,(824026,824027,824028)),
}
CONFIGS=[dict(name=f'{view}_d{depth}_t{trees}',view=view,depth=depth,trees=trees)
         for view,depth,trees in [('aux',1,100),('aux',2,50),('aux',2,100),
                                 ('aux',2,250),('aux',3,100),('aux',3,250),
                                 ('family',2,100),('family',3,250)]]
CONFIG_MAP={r['name']:r for r in CONFIGS}


def save(path,value):b.save(path,value)
def ranklog(p):return logit(np.clip((rankdata(p)-.5)/len(p),.002,.998))
def ranknorm(p):return ndtri(np.clip((rankdata(p)-.5)/len(p),.002,.998))
def score(metric,y,p):
    if metric=='MAE':return float(mean_absolute_error(y,p))
    if metric=='Spearman':return float(spearmanr(y,p).statistic)
    return float((roc_auc_score if metric=='AUROC' else average_precision_score)(y,p))
def gain(metric,y,old,new):
    s0,s1=score(metric,y,old),score(metric,y,new)
    return (s0-s1)/max(s0,1e-10) if metric=='MAE' else s1-s0
def calibrate(y,z):
    x=np.c_[np.ones(len(z)),z]
    def fun(t):
        eta=x@t;r=expit(eta)-y
        return (np.logaddexp(0,eta)-y*eta).mean()+.001*t[1]**2,x.T@r/len(r)+[0.,.002*t[1]]
    fit=minimize(fun,[0.,1.],jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B')
    assert fit.success,fit.message
    return fit.x


def aux(d,k,final=False):
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<(2 if final else 1))&(d['tid']!=k))
    assert not np.any(d['tid'][source]==k) and not np.any(d['split'][source]==2)
    known={}
    for i in source:known.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    a=np.full((len(d['y']),2*len(cols)),np.nan,np.float32)
    for i in np.flatnonzero(d['tid']==k):
        for j,t in enumerate(cols):
            v=known.get((str(d['canonical'][i]),t))
            if v:a[i,j]=np.median(v)
    a[:,len(cols):]=np.isfinite(a[:,:len(cols)])
    counts=np.sum(a[:,len(cols):],axis=1).astype(int)
    assert np.array_equal(counts,np.isfinite(a[:,:len(cols)]).sum(1))
    return a,cols,source,counts


def split(d,k,part):
    active=d['tid']==k
    if part=='final':
        fit=np.flatnonzero(active&(d['split']<2));q=np.flatnonzero(active&(d['split']==2))
    elif part=='valid':
        fit=np.flatnonzero(active&(d['split']==0));q=np.flatnonzero(active&(d['split']==1))
    else:
        ff=d['legacy_folds'];f=int(part[-1]);fit=np.flatnonzero(active&(d['split']==0)&(ff!=f));q=np.flatnonzero(active&(d['split']==0)&(ff==f))
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':assert not set(d['groups'][fit])&set(d['groups'][q])
    return fit,q


def raw_proxy(d,ix):
    p=np.where(d['split'][ix]==1,d['proxy_valid'][ix],d['proxy_cv'][ix])
    assert np.isfinite(p).all()
    return p


def features(a,cols,task,view,z):
    if view=='family':
        names=[dname for dname in [
            'cyp2c9_substrate_carbonmangels','cyp2d6_substrate_carbonmangels',
            'cyp3a4_substrate_carbonmangels','cyp2c9_veith','cyp2d6_veith','cyp3a4_veith',
            'clearance_hepatocyte_az','clearance_microsome_az','half_life_obach',
            'pgp_broccatelli','bioavailability_ma','hia_hou'] if dname!=task]
        ix=[j for j,c in enumerate(cols) if TASK_NAMES[c] in names]
        a=a[:,ix+[j+len(cols) for j in ix]]
    return np.c_[a,z].astype(np.float32)


def fit(d,k,part,cfg,seeds,cache_kind='candidate'):
    task=d['tasks'][k];metric=d['metrics'][k];iscls=metric in ('AUROC','AUPRC')
    fitrows,q=split(d,k,part);folder=OUT/cache_kind/task/part/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'predictions.npz';signature=b.sha(__file__)
    if path.exists():
        zz=np.load(path);assert str(zz['signature'])==signature
        assert zz['seeds'].tolist()==list(seeds);np.testing.assert_array_equal(zz['query'],q)
        return zz['p'].copy()
    a,cols,source,counts=aux(d,k,part=='final')
    p=raw_proxy(d,fitrows)
    if part=='final':
        hist=ROOT/'admet_submissions/v73_v72/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv'
        # Original pre-residual endpoint predictions; do not residualize an
        # already modified v82 endpoint a second time.
        rs=b.table(hist);qp=np.array([float(r['y_pred']) for r in rs if r['task']==task])
    else:qp=raw_proxy(d,q)
    y=d['y'][fitrows]
    if iscls:
        z,qz=ranklog(p),ranklog(qp);beta=calibrate(y,z)
        off=beta[0]+beta[1]*z;qoff=beta[0]+beta[1]*qz;yt=y;state=dict(beta=beta.tolist(),center=0.,scale=1.)
    elif metric=='MAE':
        scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
        center=float(np.median(y));z=(p-center)/scale;qz=(qp-center)/scale
        yt=(y-p)/scale;off=np.zeros(len(p));qoff=qp;state=dict(center=center,scale=scale)
    else:
        z,qz=ranknorm(p),ranknorm(qp);target=ranknorm(y)
        design=np.c_[np.ones(len(z)),z];beta=np.linalg.solve(design.T@design+np.diag([1e-8,1.]),design.T@target);beta[1]=max(beta[1],.01)
        yt=target-(design@beta);off=np.zeros(len(p));qoff=beta[0]+beta[1]*qz;state=dict(beta=beta.tolist(),center=0.,scale=1.)
    x=features(a[fitrows],cols,task,cfg['view'],z);qx=features(a[q],cols,task,cfg['view'],qz)
    members=[];hashes={};started=time.monotonic()
    for seed in seeds:
        kw=dict(iterations=cfg['trees'],depth=cfg['depth'],learning_rate=.025,l2_leaf_reg=50,random_seed=seed,
                thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85,task_type='GPU',devices='0')
        if iscls:
            m=CatBoostClassifier(loss_function='Logloss',**kw);m.fit(Pool(x,yt,baseline=off));member=qoff+m.predict(qx,prediction_type='RawFormulaVal')
        else:
            m=CatBoostRegressor(loss_function='MAE' if metric=='MAE' else 'RMSE',**kw);m.fit(x,yt);member=qoff+m.predict(qx)*state['scale']
        members.append(member);mp=folder/f'model_{seed}.cbm';m.save_model(str(mp));hashes[mp.name]=b.sha(mp)
    members=np.stack(members);pred=expit(members.mean(0)) if iscls else members.mean(0)
    if task=='ppbr_az':pred=pred.clip(0,100)
    assert np.isfinite(pred).all()
    np.savez_compressed(path,signature=signature,fit=fitrows,query=q,p=pred,seeds=seeds,members=members,
                        x=x,qx=qx,off=off,qoff=qoff,base_fit=p,base_query=qp,source=source,known_counts=counts[q])
    save(folder/'metadata.json',dict(signature=signature,task=task,part=part,metric=metric,config=cfg,seeds=list(seeds),
         state=state,model_hashes=hashes,predictions_sha256=b.sha(path),source_splits=[0,1] if part=='final' else [0],
         target_assay_absent=True,source_rows=len(source),ensemble='sigmoid(mean(raw_logits))' if iscls else 'mean(prediction)',seconds=time.monotonic()-started))
    return pred


def baseline(d,k,part):
    task=d['tasks'][k];q=split(d,k,part)[1]
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==task])
    if task not in BASE_SPECS:return raw_proxy(d,q)
    depth,trees,seeds=BASE_SPECS[task]
    cfg=dict(name=f'v82_original_d{depth}_t{trees}',view='aux',depth=depth,trees=trees)
    return fit(d,k,part,cfg,seeds,'baseline')


def develop(d):
    save(OUT/'protocol.json',dict(version=83,baseline='v82',baseline_official_score_user_reported=50.850,
        anchor_sha256=ANCHOR_SHA,seeds=SEEDS,configs=CONFIGS,select_parts=['fold0','fold1'],confirm_parts=['fold2','valid'],
        selection='Freeze task/config/weight on folds0/1; require both positive. Confirmation accepts only if fold2 and valid positive.',
        historical_proxy_limitations=True,historical_labels_reused=True,test_labels_used=False,external_data=False,
        mask_count='sum(known_mask), checked against finite measurement values',ensemble='classification mean raw logits at every stage',signature=b.sha(__file__)))
    records=[]
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k];parts={p:(split(d,k,p)[1],baseline(d,k,p)) for p in ('fold0','fold1')}
        for cfg in CONFIGS:
            pp={part:fit(d,k,part,cfg,SEEDS[:2]) for part in parts}
            for weight in (.25,.5,1.):
                metrics=[]
                for part,(q,old) in parts.items():
                    new=pp[part]
                    if metric!='MAE':mixed=(1-weight)*rankdata(old)/len(old)+weight*rankdata(new)/len(new)
                    else:mixed=(1-weight)*old+weight*new
                    metrics.append(dict(part=part,baseline=score(metric,d['y'][q],old),candidate=score(metric,d['y'][q],mixed),gain=gain(metric,d['y'][q],old,mixed)))
                records.append(dict(task=task,metric=metric,config=cfg,weight=weight,parts=metrics,objective=min(p['gain'] for p in metrics)))
        save(OUT/'development.json',dict(status='IN_PROGRESS',records=records,signature=b.sha(__file__)))
        best=max([r for r in records if r['task']==task],key=lambda r:r['objective'])
        print(json.dumps(dict(stage='development',task=task,best=best),ensure_ascii=False),flush=True)
    selected=[]
    for task in d['tasks']:
        choices=[r for r in records if r['task']==task];chosen=max(choices,key=lambda r:r['objective'])
        # A small practical margin limits selecting arbitrary tied AUROC ranks.
        if chosen['objective']>(.0004 if chosen['metric']!='MAE' else .002):selected.append(chosen)
    save(OUT/'development.json',dict(status='COMPLETE',records=records,signature=b.sha(__file__)))
    save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',selected=selected,
         signature=b.sha(__file__),confirmation_inspected=False,development_ensemble_seeds=list(SEEDS[:2]),
         confirmation_ensemble_seeds=list(SEEDS),test_labels_used=False))
    print(json.dumps(dict(stage='frozen',selected=selected),ensure_ascii=False),flush=True)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==b.sha(__file__)
    records=[];accepted=[]
    for chosen in frozen['selected']:
        task=chosen['task'];k=d['tasks'].index(task);metric=chosen['metric'];cfg=chosen['config'];w=chosen['weight'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=split(d,k,part)[1];old=baseline(d,k,part);new=fit(d,k,part,cfg,SEEDS,'confirmation')
            mixed=(1-w)*old+w*new if metric=='MAE' else (1-w)*rankdata(old)/len(old)+w*rankdata(new)/len(new)
            parts.append(dict(part=part,rows=len(q),baseline=score(metric,d['y'][q],old),candidate=score(metric,d['y'][q],mixed),gain=gain(metric,d['y'][q],old,mixed)))
        ok=all(p['gain']>0 for p in parts)
        record=dict(**chosen,confirmation=parts,accepted=ok);records.append(record)
        if ok:accepted.append(task)
        print(json.dumps(dict(stage='confirmation',**record),ensure_ascii=False),flush=True)
        save(OUT/'confirmation.json',dict(status='IN_PROGRESS',records=records,accepted=accepted,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    save(OUT/'confirmation.json',dict(status='COMPLETE',records=records,accepted=accepted,frozen_sha256=b.sha(OUT/'frozen_selection.json'),test_labels_used=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','confirm']);args=parser.parse_args()
    d=b.load_data();d['legacy_folds']=np.load(ROOT/'recovered_history/legacy_v24_folds.npy')
    global TASK_NAMES;TASK_NAMES=d['tasks']
    assert b.sha(ANCHOR)==ANCHOR_SHA
    (develop if args.stage=='develop' else confirm)(d)


if __name__=='__main__':main()
